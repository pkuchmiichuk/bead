// Validates JSON values against the layers lexicons using the ATProto lexicon
// validation machinery (@atproto/lexicon). Reads a JSON array of
// {lexUri, value} pairs from stdin and writes a JSON array of {ok, error?}.
import { readFileSync, readdirSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { Lexicons } from "@atproto/lexicon";

const here = dirname(fileURLToPath(import.meta.url));
const lexiconDir =
  process.env.LAYERS_LEXICON_DIR ||
  join(here, "..", "..", "..", "vendor", "layers", "lexicons", "pub", "layers");

function lexiconFiles(dir) {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...lexiconFiles(full));
    else if (entry.name.endsWith(".json")) out.push(full);
  }
  return out.sort();
}

const lexicons = new Lexicons();
for (const file of lexiconFiles(lexiconDir)) {
  lexicons.add(JSON.parse(readFileSync(file, "utf8")));
}

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => (input += chunk));
process.stdin.on("end", () => {
  const items = JSON.parse(input);
  const results = items.map(({ lexUri, value }) => {
    try {
      const result = lexicons.validate(lexUri, value);
      return result.success
        ? { ok: true }
        : { ok: false, lexUri, error: result.error?.message ?? "invalid" };
    } catch (err) {
      return { ok: false, lexUri, error: String(err?.message ?? err) };
    }
  });
  process.stdout.write(JSON.stringify(results));
});
