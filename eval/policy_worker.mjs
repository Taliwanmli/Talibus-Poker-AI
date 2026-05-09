import { spawn } from 'node:child_process'
import readline from 'node:readline'

function envFlag(name, defaultValue = false) {
  const value = process.env?.[name]
  if (value == null) return defaultValue
  const normalized = String(value).trim().toLowerCase()
  if (['1', 'true', 'yes', 'on'].includes(normalized)) return true
  if (['0', 'false', 'no', 'off'].includes(normalized)) return false
  return defaultValue
}

function envString(name, defaultValue = '') {
  const value = process.env?.[name]
  if (value == null) return defaultValue
  const normalized = String(value).trim()
  return normalized.length > 0 ? normalized : defaultValue
}

const INCLUDE_ENGINE_DEBUG = envFlag('WIPOKER_ENGINE_DEBUG', false)
const POLICY_SOURCE = envString('WIPOKER_POLICY_SOURCE', 'node').toLowerCase()

let getRecommendation = null
if (POLICY_SOURCE !== 'blueprint') {
  try {
    const engine = await import('./dist/engine/simulationDecisionEngine.js')
    getRecommendation = engine.getRecommendation
    if (typeof getRecommendation !== 'function') {
      throw new Error('getRecommendation export not found')
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    process.stderr.write(
      `[policy_worker] failed to load compiled engine. Run "npm run eval:build" first. ${message}\n`,
    )
    process.exit(1)
  }
}

let blueprintProc = null
let blueprintStderr = ''
const blueprintPending = []

function startBlueprintWorkerIfNeeded() {
  if (POLICY_SOURCE !== 'blueprint') return
  const command = envString(
    'WIPOKER_BLUEPRINT_WORKER_CMD',
    'cargo run --manifest-path solver/Cargo.toml -p player --bin blueprint_policy_worker --release',
  )

  blueprintProc = spawn(command, {
    cwd: process.cwd(),
    shell: true,
    stdio: ['pipe', 'pipe', 'pipe'],
  })

  const stdoutRl = readline.createInterface({
    input: blueprintProc.stdout,
    crlfDelay: Infinity,
    terminal: false,
  })

  stdoutRl.on('line', (line) => {
    const pending = blueprintPending.shift()
    if (!pending) return
    try {
      const payload = JSON.parse(line)
      pending.resolve(payload)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      pending.reject(new Error(`invalid JSON from blueprint worker: ${message}`))
    }
  })

  if (blueprintProc.stderr) {
    blueprintProc.stderr.on('data', (chunk) => {
      blueprintStderr += String(chunk)
      if (blueprintStderr.length > 4000) {
        blueprintStderr = blueprintStderr.slice(-4000)
      }
    })
  }

  blueprintProc.on('exit', (code) => {
    const err = new Error(
      `blueprint worker exited with code ${code ?? 'unknown'}: ${blueprintStderr.trim()}`,
    )
    while (blueprintPending.length > 0) {
      blueprintPending.shift()?.reject(err)
    }
  })
}

function queryBlueprintWorker(request) {
  return new Promise((resolve, reject) => {
    if (!blueprintProc || !blueprintProc.stdin) {
      reject(new Error('blueprint worker not initialized'))
      return
    }
    blueprintPending.push({ resolve, reject })
    blueprintProc.stdin.write(`${JSON.stringify(request)}\n`)
  })
}

function writeResponse(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`)
}

function normalizeResult(result) {
  const executedAction = result?.executedAction ?? result?.recommendedAction ?? null
  const argmaxAction = result?.argmaxAction ?? result?.recommendedAction ?? null
  return {
    status: result?.status ?? 'ok',
    recommendedAction: executedAction,
    executedAction,
    argmaxAction,
    mix: result?.mix ?? null,
    explanation: result?.explanation,
    missingFields: result?.missingFields,
    debug: result?.debug ?? null,
  }
}

startBlueprintWorkerIfNeeded()

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
  terminal: false,
})

rl.on('line', async (line) => {
  const trimmed = line.trim()
  if (!trimmed) return

  let input
  try {
    input = JSON.parse(trimmed)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    writeResponse({
      status: 'unavailable',
      recommendedAction: null,
      mix: null,
      error: `invalid_json: ${message}`,
    })
    return
  }

  const street = input?.street
  const handState = input?.handState
  if (
    !street ||
    !handState ||
    (street !== 'preflop' &&
      street !== 'flop' &&
      street !== 'turn' &&
      street !== 'river')
  ) {
    writeResponse({
      status: 'unavailable',
      recommendedAction: null,
      mix: null,
      error: 'invalid_payload: expected { street, handState }',
    })
    return
  }

  try {
    let rawResult
    if (POLICY_SOURCE === 'blueprint') {
      rawResult = await queryBlueprintWorker({ street, handState })
    } else {
      rawResult = getRecommendation(street, handState, { debug: INCLUDE_ENGINE_DEBUG })
    }

    const normalized = normalizeResult(rawResult)
    const payload = {
      status: normalized.status,
      recommendedAction: normalized.recommendedAction,
      executedAction: normalized.executedAction,
      argmaxAction: normalized.argmaxAction,
      mix: normalized.mix,
    }
    if (INCLUDE_ENGINE_DEBUG) {
      payload.explanation = normalized.explanation
      payload.missingFields = normalized.missingFields
      payload.debug = normalized.debug
    }
    writeResponse(payload)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    writeResponse({
      status: 'unavailable',
      recommendedAction: null,
      mix: null,
      error: `engine_error: ${message}`,
    })
  }
})

rl.on('close', () => {
  if (blueprintProc && blueprintProc.exitCode === null) {
    blueprintProc.kill()
  }
  process.exit(0)
})
