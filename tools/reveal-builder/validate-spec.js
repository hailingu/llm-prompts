const fs = require('fs');
const path = require('path');
const Ajv = require('ajv');

const schemaPath = path.resolve(__dirname, '../../docs/specs/design-spec.reveal.schema.json');
const specPath = path.resolve(__dirname, '../../docs/specs/design-spec.reveal.json');

const schema = JSON.parse(fs.readFileSync(schemaPath, 'utf8'));
const spec = JSON.parse(fs.readFileSync(specPath, 'utf8'));

const ajv = new Ajv({allErrors: true, strict: false});
const validate = ajv.compile(schema);
const valid = validate(spec);
if (!valid) {
  console.error('Validation failed. Errors:');
  console.error(validate.errors);
  process.exit(2);
} else {
  console.log('Validation successful: design-spec.reveal.json conforms to schema.');
}
