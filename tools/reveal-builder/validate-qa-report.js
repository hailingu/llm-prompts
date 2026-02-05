const fs = require('fs');
const path = require('path');
const Ajv = require('ajv');

const schemaPath = path.resolve(__dirname, '../../docs/specs/qa-report.schema.json');
const samplePath = path.resolve(__dirname, '../../docs/specs/sample-qa-report.json');

const schema = JSON.parse(fs.readFileSync(schemaPath, 'utf8'));
const sample = JSON.parse(fs.readFileSync(samplePath, 'utf8'));

const ajv = new Ajv({allErrors: true, strict: false});
const validate = ajv.compile(schema);
const valid = validate(sample);
if (!valid) {
  console.error('QA Report validation failed. Errors:');
  console.error(validate.errors);
  process.exit(2);
} else {
  console.log('Validation successful: sample-qa-report.json conforms to qa-report.schema.json');
}
