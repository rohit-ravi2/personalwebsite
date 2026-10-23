/** Interpolate saved vectors using numeric keys (Python exports 0.0, JS uses 0). */
export function interpolateDeconv(lookup, cancer, fraction) {
  const entries = lookup.lookup[cancer] ?? lookup.lookup.healthy;
  const points = Object.entries(entries).map(([key, vector]) => [Number(key), vector]).filter(([x]) => Number.isFinite(x)).sort((a, b) => a[0] - b[0]);
  if (!points.length) throw new Error('No mixture data are available.');
  if (fraction <= points[0][0]) return [...points[0][1]];
  if (fraction >= points.at(-1)[0]) return [...points.at(-1)[1]];
  const upper = points.findIndex(([x]) => x >= fraction);
  const [lo, a] = points[upper - 1];
  const [hi, b] = points[upper];
  const alpha = (fraction - lo) / (hi - lo);
  return a.map((value, i) => value + alpha * (b[i] - value));
}
