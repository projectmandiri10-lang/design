import test from 'node:test';
import assert from 'node:assert/strict';
import {
  assertSafeAssetReference,
  parseColorCount,
} from '../src/utils/vectorizeValidation.js';

test('parseColorCount accepts 2 to 6', () => {
  assert.equal(parseColorCount('2'), 2);
  assert.equal(parseColorCount('6'), 6);
});

test('parseColorCount rejects invalid values', () => {
  assert.throws(() => parseColorCount('1'));
  assert.throws(() => parseColorCount('7'));
  assert.throws(() => parseColorCount('abc'));
});

test('assertSafeAssetReference rejects traversal', () => {
  assert.throws(() => assertSafeAssetReference('../bad', 'file.svg'));
  assert.throws(() => assertSafeAssetReference('job', '../../file.svg'));
});
