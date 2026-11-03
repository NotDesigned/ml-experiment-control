//! A single read transaction fingerprints schema and all rows, including
//! original event IDs and W&B cursors. Never copies or checkpoints a database.
use super::*;
use rusqlite::types::ValueRef;
use sha2::{Digest, Sha256};
const MAX_ROWS: usize = 2_000_000;
const MAX_BYTES: usize = 1024 * 1024 * 1024;
fn name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_')
}
pub(super) fn capture(
    root: &Directory,
    parent: &str,
    leaf: &str,
    label: &str,
    audit: &mut Audit,
) -> Result<()> {
    match optional_child(root, parent)? {
        Some(directory) => capture_direct(&directory, leaf, label, audit),
        None => audit.push(
            key("native-state", None, None, "database", Some(label)),
            &Value::Null,
            0,
        ),
    }
}
pub(super) fn capture_direct(
    directory: &Directory,
    leaf: &str,
    label: &str,
    audit: &mut Audit,
) -> Result<()> {
    let k = key("native-state", None, None, "database", Some(label));
    if !directory
        .entry_exists(OsStr::new(leaf))
        .map_err(|_| Error::Evidence)?
    {
        return audit.push(k, &Value::Null, 0);
    }
    let database =
        crate::storage::SqliteDatabase::in_directory(directory.clone(), OsStr::new(leaf))
            .map_err(|_| Error::Evidence)?;
    let tables = database.read_only(|connection| -> std::result::Result<_, crate::storage::Error> {
        connection.execute_batch("BEGIN DEFERRED TRANSACTION")?;
        let user_version: i64 = connection.query_row("PRAGMA user_version",[],|row|row.get(0))?;
        let application_id: i64 = connection.query_row("PRAGMA application_id",[],|row|row.get(0))?;
        let (schema_rows,schema_bytes):(i64,i64)=connection.query_row("SELECT COUNT(*),COALESCE(SUM(length(CAST(type AS BLOB))+length(CAST(name AS BLOB))+length(CAST(tbl_name AS BLOB))+COALESCE(length(CAST(sql AS BLOB)),0)),0) FROM sqlite_schema",[],|row|Ok((row.get(0)?,row.get(1)?)))?;
        if !(0..=1024).contains(&schema_rows) || !(0..=8_388_608).contains(&schema_bytes) { return Err(crate::storage::Error::Invalid("audit schema budget")); }
        let mut schema = connection.prepare("SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name")?;
        let schema = schema.query_map([], |row| Ok(json!([row.get::<_,String>(0)?,row.get::<_,String>(1)?,row.get::<_,String>(2)?,row.get::<_,Option<String>>(3)?])))?.take(1025).collect::<std::result::Result<Vec<_>,_>>()?;
        if schema.len()>1024 { return Err(crate::storage::Error::Invalid("audit schema budget")); }
        let mut statement = connection.prepare("SELECT name,sql FROM sqlite_schema WHERE type='table' ORDER BY name")?;
        let tables = statement.query_map([], |row| Ok((row.get::<_,String>(0)?, row.get::<_,String>(1)?)))?.collect::<std::result::Result<Vec<_>,_>>()?;
        if tables.len() > 128 { return Err(crate::storage::Error::Invalid("audit table budget")); }
        let mut total_rows = 0usize;
        let mut total_bytes = 0usize;
        let mut result = Vec::new();
        for (table, sql) in tables {
            if !name(&table) || sql.len() > 1024 * 1024 { return Err(crate::storage::Error::Invalid("audit table shape")); }
            let mut rows = connection.prepare(&format!("SELECT * FROM \"{table}\""))?;
            let columns = rows.column_names().iter().map(|v|v.to_string()).collect::<Vec<_>>();
            let mut cursor = rows.query([])?;
            let mut hashes = Vec::new();
            while let Some(row) = cursor.next()? {
                total_rows = total_rows.checked_add(1).filter(|v|*v <= MAX_ROWS).ok_or(crate::storage::Error::Invalid("audit row budget"))?;
                let mut hash = Sha256::new();
                let mut row_bytes = 0usize;
                for column in 0..columns.len() {
                    let value = row.get_ref(column)?;
                    let (tag, bytes) = match value {
                        ValueRef::Null => (0u8, Vec::new()),
                        ValueRef::Integer(v) => (1, v.to_le_bytes().to_vec()),
                        ValueRef::Real(v) => (2, v.to_bits().to_le_bytes().to_vec()),
                        ValueRef::Text(v) | ValueRef::Blob(v) => {
                            if v.len() > 8 * 1024 * 1024 { return Err(crate::storage::Error::Invalid("audit cell budget")); }
                            (if matches!(value, ValueRef::Text(_)) {3} else {4}, v.to_vec())
                        }
                    };
                    row_bytes = row_bytes.checked_add(bytes.len()).filter(|v|*v <= 8 * 1024 * 1024).ok_or(crate::storage::Error::Invalid("audit row size"))?;
                    hash.update([tag]); hash.update((bytes.len() as u64).to_le_bytes()); hash.update(bytes);
                }
                total_bytes = total_bytes.checked_add(row_bytes).filter(|v|*v <= MAX_BYTES).ok_or(crate::storage::Error::Invalid("audit byte budget"))?;
                hashes.push(<[u8;32]>::from(hash.finalize()));
            }
            hashes.sort_unstable();
            let mut hash = Sha256::new();
            for row in &hashes { hash.update(row); }
            result.push(json!({"table":table,"sql":sql,"columns":columns,"rows":hashes.len(),"rows_sha256":format!("sha256:{:x}",hash.finalize())}));
        }
        connection.execute_batch("ROLLBACK")?;
        Ok((json!({"user_version":user_version,"application_id":application_id,"schema":schema,"tables":result}),total_rows))
    }).map_err(|error| match error {
        crate::storage::Error::Invalid("audit byte budget") => Error::DatabaseBytes,
        crate::storage::Error::Invalid("audit row budget") => Error::DatabaseRows,
        crate::storage::Error::Invalid("audit schema budget" | "audit table budget" | "audit table shape") => Error::DatabaseShape,
        crate::storage::Error::Invalid("audit cell budget" | "audit row size") => Error::DatabaseValue,
        crate::storage::Error::Detached | crate::storage::Error::Missing | crate::storage::Error::Io(_) => Error::DatabaseBinding,
        crate::storage::Error::Sqlite(rusqlite::Error::InvalidColumnType(..) | rusqlite::Error::InvalidColumnIndex(_) | rusqlite::Error::FromSqlConversionFailure(..)) => Error::DatabaseSqlValue,
        crate::storage::Error::Sqlite(rusqlite::Error::SqliteFailure(code, _))
            if matches!(code.code, rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked) => Error::DatabaseBusy,
        crate::storage::Error::Sqlite(rusqlite::Error::SqliteFailure(code, _))
            if matches!(code.code, rusqlite::ErrorCode::ReadOnly | rusqlite::ErrorCode::CannotOpen | rusqlite::ErrorCode::PermissionDenied) => Error::DatabaseReadOnly,
        _ => Error::Database,
    })?;
    audit.push(k, &json!(tables.0), tables.1 as u64)
}
