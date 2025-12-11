-- name: create-user-table
CREATE TABLE users AS SELECT * FROM read_json_auto('./testdata/json_test/input_files/input_users.jsonl', auto_detect=true, records=true, sample_size=-1)

-- name: add-typed-columns
ALTER TABLE users ADD name VARCHAR;
ALTER TABLE users ADD age INTEGER;
ALTER TABLE users ADD height FLOAT;
ALTER TABLE users ADD awesome BOOLEAN;
ALTER TABLE users ADD bday DATE;

-- name: cast-raw-columns
UPDATE users SET name = name_raw.first || ' ' || name_raw.last;
UPDATE users SET age = CASE WHEN regexp_full_match(age_raw,' -?\d+') THEN CAST(age_raw as INTEGER) ELSE 0 END;
UPDATE users SET height = CASE WHEN regexp_full_match(height_raw,' (\+|\-)?(\d+).(\d+)') THEN CAST(height_raw as FLOAT) ELSE 0 END;
UPDATE users SET awesome = CASE WHEN regexp_full_match(awesome_raw,' (true|false)', 'i') THEN CAST (TRIM(awesome_raw) as BOOLEAN) ELSE false END;
UPDATE users SET bday = CASE WHEN regexp_full_match(bday_raw,' (\d+)-(\d+)-(\d+)') THEN CAST(bday_raw as DATE) ELSE '2006-01-02' END;

-- name: add-mapped-columns
ALTER TABLE users ADD name_mapped VARCHAR;
ALTER TABLE users ADD age_mapped INTEGER;
ALTER TABLE users ADD height_mapped FLOAT;
ALTER TABLE users ADD awesome_mapped BOOLEAN;
ALTER TABLE users ADD bday_mapped DATE;

-- name: set-mapped-columns
UPDATE users SET name_mapped = name;
UPDATE users SET age_mapped = age;
UPDATE users SET height_mapped = height;
UPDATE users SET awesome_mapped = awesome;
UPDATE users SET bday_mapped = bday;

-- name: export-to-csv
COPY (
  SELECT name_mapped as name_clever,
  age_mapped as age_clever,
  height_mapped as height_clever,
  awesome_mapped as awesome_clever,
  bday_mapped as bday_clever
  FROM users
) TO './testdata/json_test/output_files/output_users.csv' (HEADER, DELIMITER ',');
