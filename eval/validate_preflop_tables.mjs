/**
 * Preflop Table Validator
 *
 * Validates structural integrity of the GTO preflop frequency tables.
 * Each scenario must have exactly 169 hand classes, each with raise/call/fold
 * frequencies in [0,1] summing to ~1.0. Default scenarios must have aggregate
 * continue rates within expected GTO bounds.
 *
 * Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
 */

import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const TABLES_PATH = resolve(__dirname, '..', 'src', 'data', 'poker_trainer_preflop_tables_extracted.json')

const EXPECTED_HAND_CLASSES = 169

const DEFAULT_OOP_SCENARIO = 'facing_raise__hero_BB__villain_BTN'
const DEFAULT_IP_SCENARIO = 'unopened_pot__hero_BTN'

// Expected aggregate continue rate bounds
const OOP_CONTINUE_MIN = 0.40
const OOP_CONTINUE_MAX = 0.48
const IP_CONTINUE_MIN = 0.35
const IP_CONTINUE_MAX = 0.55

/**
 * Validate the preflop GTO frequency tables for structural correctness.
 *
 * @param {Record<string, Record<string, {raise: number, call: number, fold: number}>>} tables
 *   The tables object from the preflop JSON (keyed by scenario, then hand class).
 * @returns {{ valid: boolean, errors: string[] }}
 */
export function validatePreflopTables(tables) {
  const errors = []

  if (!tables || typeof tables !== 'object') {
    return { valid: false, errors: ['tables is null or not an object'] }
  }

  const scenarioKeys = Object.keys(tables)
  if (scenarioKeys.length === 0) {
    return { valid: false, errors: ['tables has no scenarios'] }
  }

  for (const scenario of scenarioKeys) {
    const table = tables[scenario]
    const handClasses = Object.keys(table)

    // Requirement 2.1: Each scenario must have exactly 169 hand class entries
    if (handClasses.length !== EXPECTED_HAND_CLASSES) {
      errors.push(
        `${scenario}: expected ${EXPECTED_HAND_CLASSES} hand classes, got ${handClasses.length}`
      )
    }

    // Requirement 2.2: Each hand class has raise/call/fold in [0,1] summing to [0.99, 1.01]
    for (const hand of handClasses) {
      const mix = table[hand]
      const raise = mix.raise
      const call = mix.call
      const fold = mix.fold

      // Check each frequency is in [0, 1]
      for (const [name, val] of [['raise', raise], ['call', call], ['fold', fold]]) {
        if (typeof val !== 'number' || !isFinite(val)) {
          errors.push(`${scenario}/${hand}: ${name} is not a finite number (${val})`)
        } else if (val < 0 || val > 1) {
          errors.push(`${scenario}/${hand}: ${name} = ${val} is outside [0, 1]`)
        }
      }

      // Check sum ≈ 1.0
      const sum = (raise || 0) + (call || 0) + (fold || 0)
      if (sum < 0.99 || sum > 1.01) {
        errors.push(
          `${scenario}/${hand}: raise+call+fold = ${sum.toFixed(4)}, expected [0.99, 1.01]`
        )
      }
    }
  }

  // Requirement 2.3: OOP aggregate continue rate in [0.40, 0.48]
  if (tables[DEFAULT_OOP_SCENARIO]) {
    const rate = aggregateContinueRate(tables[DEFAULT_OOP_SCENARIO])
    if (rate < OOP_CONTINUE_MIN || rate > OOP_CONTINUE_MAX) {
      errors.push(
        `${DEFAULT_OOP_SCENARIO}: aggregate continue rate = ${rate.toFixed(4)}, expected [${OOP_CONTINUE_MIN}, ${OOP_CONTINUE_MAX}]`
      )
    }
  } else {
    errors.push(`Missing default OOP scenario: ${DEFAULT_OOP_SCENARIO}`)
  }

  // Requirement 2.4: IP aggregate continue rate in [0.35, 0.55]
  if (tables[DEFAULT_IP_SCENARIO]) {
    const rate = aggregateContinueRate(tables[DEFAULT_IP_SCENARIO])
    if (rate < IP_CONTINUE_MIN || rate > IP_CONTINUE_MAX) {
      errors.push(
        `${DEFAULT_IP_SCENARIO}: aggregate continue rate = ${rate.toFixed(4)}, expected [${IP_CONTINUE_MIN}, ${IP_CONTINUE_MAX}]`
      )
    }
  } else {
    errors.push(`Missing default IP scenario: ${DEFAULT_IP_SCENARIO}`)
  }

  return { valid: errors.length === 0, errors }
}

/**
 * Number of specific card combos for a hand class.
 * Pairs (e.g. "AA") = C(4,2) = 6, suited (e.g. "AKs") = 4, offsuit (e.g. "AKo") = 12.
 */
function comboCount(handClass) {
  if (handClass.length === 2) return 6   // pair
  if (handClass.endsWith('s')) return 4  // suited
  return 12                              // offsuit
}

/**
 * Compute the aggregate continue rate for a scenario table.
 * Continue_Weight = raise + call for each hand class.
 * Aggregate = combo-weighted average across all hand classes, where each class
 * is weighted by its number of specific combos (pairs=6, suited=4, offsuit=12).
 * This produces the true frequency of hands continuing to the flop.
 *
 * @param {Record<string, {raise: number, call: number, fold: number}>} table
 * @returns {number}
 */
function aggregateContinueRate(table) {
  const hands = Object.keys(table)
  if (hands.length === 0) return 0
  let weightedTotal = 0
  let totalCombos = 0
  for (const hand of hands) {
    const mix = table[hand]
    const cw = (mix.raise || 0) + (mix.call || 0)
    const nc = comboCount(hand)
    weightedTotal += cw * nc
    totalCombos += nc
  }
  return totalCombos > 0 ? weightedTotal / totalCombos : 0
}

/**
 * Load the preflop tables from disk and validate them.
 * Convenience entry point for subprocess invocation.
 *
 * @returns {{ valid: boolean, errors: string[] }}
 */
export function loadAndValidate() {
  const raw = readFileSync(TABLES_PATH, 'utf-8')
  const data = JSON.parse(raw)
  return validatePreflopTables(data.tables)
}

// CLI entry point: node eval/validate_preflop_tables.mjs
// Prints JSON result to stdout, exits non-zero on validation failure.
const isMain = process.argv[1] &&
  resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))

if (isMain) {
  const result = loadAndValidate()
  console.log(JSON.stringify(result, null, 2))
  if (!result.valid) {
    console.error(`Preflop table validation FAILED with ${result.errors.length} error(s)`)
    process.exit(1)
  } else {
    console.error('Preflop table validation PASSED')
  }
}
