import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const distRoot = path.join(__dirname, 'dist')

async function walk(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true })
  const files = []
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      files.push(...(await walk(fullPath)))
      continue
    }
    if (entry.isFile() && entry.name.endsWith('.js')) {
      files.push(fullPath)
    }
  }
  return files
}

function ensureJsExtension(specifier) {
  if (
    specifier.endsWith('.js') ||
    specifier.endsWith('.mjs') ||
    specifier.endsWith('.cjs') ||
    specifier.endsWith('.json') ||
    specifier.endsWith('.node')
  ) {
    return specifier
  }
  return `${specifier}.js`
}

function patchSource(code) {
  let next = code

  next = next.replace(
    /(from\s+['"])(\.\.?\/[^'"]+)(['"])/g,
    (_match, prefix, specifier, suffix) => `${prefix}${ensureJsExtension(specifier)}${suffix}`,
  )

  next = next.replace(
    /(import\s*\(\s*['"])(\.\.?\/[^'"]+)(['"]\s*\))/g,
    (_match, prefix, specifier, suffix) => `${prefix}${ensureJsExtension(specifier)}${suffix}`,
  )

  next = next.replace(
    /(from\s+['"][^'"]+\.json['"])(?!\s+with\s+\{\s*type\s*:\s*['"]json['"]\s*\})/g,
    "$1 with { type: 'json' }",
  )

  return next
}

async function main() {
  try {
    await fs.access(distRoot)
  } catch {
    console.error(`[fix_eval_dist] dist directory not found: ${distRoot}`)
    process.exit(1)
  }

  const jsFiles = await walk(distRoot)
  let updated = 0

  for (const filePath of jsFiles) {
    const source = await fs.readFile(filePath, 'utf8')
    const patched = patchSource(source)
    if (patched !== source) {
      await fs.writeFile(filePath, patched, 'utf8')
      updated += 1
    }
  }

  console.log(`[fix_eval_dist] patched ${updated}/${jsFiles.length} files`)
}

main().catch((error) => {
  console.error('[fix_eval_dist] failed:', error)
  process.exit(1)
})
