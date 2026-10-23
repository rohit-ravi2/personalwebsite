import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { interpolateDeconv } from '../src/lib/deconvolution.mjs';
const lookup = JSON.parse(readFileSync(new URL('../public/models/cfdna-cancer-detection/deconv_lookup.json', import.meta.url)));
test('mixture interpolation handles exported decimal keys and every cancer grid', () => {
  for (const cancer of lookup.cancer_types) {
    for (const fraction of lookup.tumor_fractions) {
      const expected = Object.entries(lookup.lookup[cancer]).find(([key]) => Number(key) === fraction)[1];
      assert.deepEqual(interpolateDeconv(lookup, cancer, fraction), expected);
    }
    const fraction = 0.003;
    const vector = interpolateDeconv(lookup, cancer, fraction);
    const low = lookup.lookup[cancer]['0.001'], high = lookup.lookup[cancer]['0.005'];
    vector.forEach((value, i) => assert.ok(Math.abs(value - (low[i] + high[i]) / 2) < 1e-12));
    assert.deepEqual(interpolateDeconv(lookup, cancer, -1), interpolateDeconv(lookup, cancer, 0));
    assert.deepEqual(interpolateDeconv(lookup, cancer, 2), interpolateDeconv(lookup, cancer, 0.2));
  }
});
