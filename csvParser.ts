/**
 * Lightweight, robust utility to parse CSV strings into array of objects.
 * Automatically casts booleans, numbers, nulls, and strings.
 */
export function parseCSV(csvString: string): Record<string, any>[] {
  const lines = csvString.trim().split("\n");
  if (lines.length < 2) return [];

  const headers = lines[0].split(",").map((h) => h.trim());
  const data: Record<string, any>[] = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    // Standard CSV splitter
    const values: string[] = [];
    let current = "";
    let inQuotes = false;

    for (let j = 0; j < line.length; j++) {
      const char = line[j];
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === "," && !inQuotes) {
        values.push(current.trim());
        current = "";
      } else {
        current += char;
      }
    }
    values.push(current.trim());

    if (values.length !== headers.length) continue;

    const row: Record<string, any> = {};
    for (let j = 0; j < headers.length; j++) {
      const header = headers[j];
      const value = values[j];

      if (value === "" || value === undefined || value === "null" || value === "NULL") {
        row[header] = null;
      } else if (value === "Yes" || value === "yes" || value === "true" || value === "TRUE") {
        row[header] = true;
      } else if (value === "No" || value === "no" || value === "false" || value === "FALSE") {
        row[header] = false;
      } else if (!isNaN(Number(value)) && value !== "") {
        row[header] = Number(value);
      } else {
        row[header] = value;
      }
    }
    data.push(row);
  }
  return data;
}

/**
 * Calculates basic descriptive statistics for numerical fields in a dataset.
 */
export function getDescriptiveStats(data: Record<string, any>[], field: string) {
  const values = data
    .map((row) => Number(row[field]))
    .filter((val) => !isNaN(val) && val !== null && val !== undefined);

  if (values.length === 0) return null;

  const sum = values.reduce((a, b) => a + b, 0);
  const mean = sum / values.length;
  const sorted = [...values].sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const median = sorted[Math.floor(sorted.length / 2)];

  const sqDiffs = values.map((v) => Math.pow(v - mean, 2));
  const variance = sqDiffs.reduce((a, b) => a + b, 0) / values.length;
  const stdDev = Math.sqrt(variance);

  return {
    count: values.length,
    sum,
    mean,
    min,
    max,
    median,
    stdDev,
  };
}
