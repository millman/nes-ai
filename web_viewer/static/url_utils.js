// URL helpers for ids parameters.

function normalizeIdsValues(values) {
  const normalized = [];
  const seen = new Set();
  (values || []).forEach((value) => {
    if (typeof value !== "string") return;
    value.split(",").forEach((part) => {
      const trimmed = part.trim();
      if (!trimmed || seen.has(trimmed)) return;
      seen.add(trimmed);
      normalized.push(trimmed);
    });
  });
  return normalized;
}

function parseIdsFromSearch(search) {
  const params = new URLSearchParams(search || "");
  return normalizeIdsValues(params.getAll("ids"));
}

function decodeIdsCommas(urlString) {
  return urlString.replace(/([?&]ids=)([^&]*)/, (match, prefix, value) => {
    return `${prefix}${value.replace(/%2C/gi, ",")}`;
  });
}

function applyIdsToUrl(url, ids) {
  const normalized = normalizeIdsValues(ids);
  url.searchParams.delete("ids");
  if (normalized.length > 0) {
    url.searchParams.set("ids", normalized.join(","));
  }
  return decodeIdsCommas(url.toString());
}

function normalizeIdsInUrl(url) {
  const ids = normalizeIdsValues(url.searchParams.getAll("ids"));
  return applyIdsToUrl(url, ids);
}
