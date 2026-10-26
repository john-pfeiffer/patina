import { describe, expect, it } from 'vitest';
import conf from '../../src-tauri/tauri.conf.json';

// Tone.js runs its transport clock in a Web Worker created from a blob: URL.
// A policy without `worker-src blob:` leaves the transport position advancing
// while nothing scheduled on it ever fires (silent playback, tracker #35).

const directive = (policy: string, name: string): string[] =>
  policy
    .split(';')
    .map((d) => d.trim())
    .filter((d) => d.startsWith(`${name} `))
    .flatMap((d) => d.split(/\s+/).slice(1));

describe('tauri content security policy', () => {
  it.each(['csp', 'devCsp'] as const)('%s allows blob: workers', (key) => {
    const policy = conf.app.security[key];
    expect(directive(policy, 'worker-src')).toContain('blob:');
    // Older WebKit resolves workers through child-src.
    expect(directive(policy, 'child-src')).toContain('blob:');
  });

  it.each(['csp', 'devCsp'] as const)('%s still lets the UI reach the sidecar', (key) => {
    const policy = conf.app.security[key];
    expect(directive(policy, 'connect-src')).toContain('http://127.0.0.1:*');
  });
});
