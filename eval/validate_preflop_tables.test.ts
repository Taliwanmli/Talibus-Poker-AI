import { describe, it, expect } from 'vitest'
import { validatePreflopTables, loadAndValidate } from './validate_preflop_tables.mjs'
import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const TABLES_PATH = resolve(__dirname, '..', 'src', 'data', 'poker_trainer_preflop_tables_extracted.json')

/**
 * Build a valid 169-hand scenario table where every hand has raise+call+fold = 1.0.
 * Uses a simple 40% raise / 20% call / 40% fold split for all hands.
 */
function buildValidScenarioTable(
  raiseFreq = 0.4,
  callFreq = 0.2,
  foldFreq = 0.4
): Record<string, { raise: number; call: number; fold: number }> {
  const ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
  const table: Record<string, { raise: number; call: number; fold: number }> = {}
  for (let i = 0; i < ranks.length; i++) {
    for (let j = 0; j < ranks.length; j++) {
      let hand: string
      if (i === j) hand = ranks[i] + ranks[j]           // pair
      else if (i < j) hand = ranks[i] + ranks[j] + 's'  // suited (higher rank first)
      else hand = ranks[j] + ranks[i] + 'o'              // offsuit
      table[hand] = { raise: raiseFreq, call: callFreq, fold: foldFreq }
    }
  }
  return table
}

/** All 169 hand class keys */
function allHandClassKeys(): string[] {
  return Object.keys(buildValidScenarioTable())
}

describe('validatePreflopTables — real data', () => {
  it('passes all checks against the real preflop tables JSON', () => {
    const result = loadAndValidate()
    expect(result.valid).toBe(true)
    expect(result.errors).toEqual([])
  })
})

describe('validatePreflopTables — structural checks', () => {
  it('fails when a scenario is missing a hand class (168 instead of 169)', () => {
    const table = buildValidScenarioTable()
    // Remove one hand class
    const firstKey = Object.keys(table)[0]
    delete table[firstKey]
    expect(Object.keys(table).length).toBe(168)

    const tables = { some_scenario: table }
    const result = validatePreflopTables(tables)
    expect(result.valid).toBe(false)
    expect(result.errors.some(e => e.includes('168') && e.includes('169'))).toBe(true)
  })

  it('fails when frequencies sum to 0.5 instead of ~1.0', () => {
    const table = buildValidScenarioTable(0.2, 0.1, 0.2) // sum = 0.5
    const tables = { some_scenario: table }
    const result = validatePreflopTables(tables)
    expect(result.valid).toBe(false)
    // Should report sum violation for every hand
    const sumErrors = result.errors.filter(e => e.includes('raise+call+fold'))
    expect(sumErrors.length).toBeGreaterThan(0)
  })

  it('fails when a frequency is outside [0, 1]', () => {
    const table = buildValidScenarioTable()
    const firstKey = Object.keys(table)[0]
    table[firstKey] = { raise: 1.5, call: 0, fold: -0.5 }
    const tables = { some_scenario: table }
    const result = validatePreflopTables(tables)
    expect(result.valid).toBe(false)
    expect(result.errors.some(e => e.includes('outside [0, 1]'))).toBe(true)
  })

  it('returns errors with scenario and hand details (Req 2.5)', () => {
    const table = buildValidScenarioTable()
    const handKey = Object.keys(table)[5]
    table[handKey] = { raise: 0.5, call: 0.5, fold: 0.5 } // sum = 1.5
    const tables = { my_test_scenario: table }
    const result = validatePreflopTables(tables)
    expect(result.valid).toBe(false)
    expect(result.errors.some(e => e.includes('my_test_scenario') && e.includes(handKey))).toBe(true)
  })
})

describe('validatePreflopTables — aggregate continue rate bounds', () => {
  it('fails when OOP scenario continue rate is outside [0.40, 0.48]', () => {
    // Build a table where continue rate (raise+call) is very high (~0.9)
    const oopTable = buildValidScenarioTable(0.7, 0.2, 0.1) // continue = 0.9
    const ipTable = buildValidScenarioTable(0.3, 0.15, 0.55) // continue = 0.45 (within IP bounds)
    const tables = {
      facing_raise__hero_BB__villain_BTN: oopTable,
      unopened_pot__hero_BTN: ipTable,
    }
    const result = validatePreflopTables(tables)
    expect(result.valid).toBe(false)
    expect(result.errors.some(e =>
      e.includes('facing_raise__hero_BB__villain_BTN') && e.includes('aggregate continue rate')
    )).toBe(true)
  })

  it('fails when IP scenario continue rate is outside [0.35, 0.55]', () => {
    // Build a table where IP continue rate is very low (~0.1)
    const oopTable = buildValidScenarioTable(0.25, 0.19, 0.56) // continue = 0.44 (within OOP bounds)
    const ipTable = buildValidScenarioTable(0.05, 0.05, 0.9)   // continue = 0.1
    const tables = {
      facing_raise__hero_BB__villain_BTN: oopTable,
      unopened_pot__hero_BTN: ipTable,
    }
    const result = validatePreflopTables(tables)
    expect(result.valid).toBe(false)
    expect(result.errors.some(e =>
      e.includes('unopened_pot__hero_BTN') && e.includes('aggregate continue rate')
    )).toBe(true)
  })

  it('passes when both default scenarios have continue rates within bounds', () => {
    const oopTable = buildValidScenarioTable(0.25, 0.19, 0.56) // continue = 0.44
    const ipTable = buildValidScenarioTable(0.3, 0.15, 0.55)   // continue = 0.45
    const tables = {
      facing_raise__hero_BB__villain_BTN: oopTable,
      unopened_pot__hero_BTN: ipTable,
    }
    const result = validatePreflopTables(tables)
    expect(result.valid).toBe(true)
    expect(result.errors).toEqual([])
  })

  it('reports missing default OOP scenario', () => {
    const ipTable = buildValidScenarioTable(0.3, 0.15, 0.55)
    const tables = { unopened_pot__hero_BTN: ipTable }
    const result = validatePreflopTables(tables)
    expect(result.valid).toBe(false)
    expect(result.errors.some(e => e.includes('Missing default OOP scenario'))).toBe(true)
  })

  it('reports missing default IP scenario', () => {
    const oopTable = buildValidScenarioTable(0.25, 0.19, 0.56)
    const tables = { facing_raise__hero_BB__villain_BTN: oopTable }
    const result = validatePreflopTables(tables)
    expect(result.valid).toBe(false)
    expect(result.errors.some(e => e.includes('Missing default IP scenario'))).toBe(true)
  })
})

describe('validatePreflopTables — edge cases', () => {
  it('rejects null input', () => {
    const result = validatePreflopTables(null as any)
    expect(result.valid).toBe(false)
    expect(result.errors[0]).toContain('null')
  })

  it('rejects empty tables object', () => {
    const result = validatePreflopTables({})
    expect(result.valid).toBe(false)
    expect(result.errors[0]).toContain('no scenarios')
  })
})
