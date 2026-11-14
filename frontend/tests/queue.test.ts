import test from "node:test";
import assert from "node:assert/strict";
import { indexedDB, IDBKeyRange } from "fake-indexeddb";

// Attach fake-indexeddb to globalThis for node test environment
(globalThis as any).indexedDB = indexedDB;
(globalThis as any).IDBKeyRange = IDBKeyRange;

import {
  enqueueOfflineScreening,
  getQueuedScreenings,
  removeQueuedScreening,
  syncQueueWithServer,
  OfflineScreeningItem,
} from "../src/offline/queue.js";

test("Queue operations: enqueue and store only pseudonymous IDs", async () => {
  const item = await enqueueOfflineScreening({
    patient_pseudonym_id: "PAT-ANON-XYZ-123",
    eye_side: "OD",
    camera_type: "TABLETOP",
    age: 52,
    diabetes_type: "TYPE_2",
    language: "en",
    image_b64: "dGVzdC1pbWFnZS1iYXNlNjQ=",
    offline_result: {
      referral: "REFERABLE",
      urgency: "PRIORITY",
      grade: 2,
      is_provisional: true,
    },
  });

  assert.ok(item.id.startsWith("offline-"));
  assert.equal(item.patient_pseudonym_id, "PAT-ANON-XYZ-123");
  assert.equal(item.status, "pending");
  assert.equal(item.retry_count, 0);

  // PHI check: ensure no patient name or direct identifiers are stored
  assert.equal((item as any).patient_name, undefined);

  const queued = await getQueuedScreenings();
  assert.ok(queued.some((q) => q.id === item.id));

  // Clean up
  await removeQueuedScreening(item.id);
});

test("Queue deletion after confirmed synchronization (HTTP 200)", async () => {
  const item = await enqueueOfflineScreening({
    patient_pseudonym_id: "PAT-SYNC-SUCCESS-001",
    eye_side: "OS",
    camera_type: "TABLETOP",
    age: 48,
    diabetes_type: "TYPE_2",
    language: "en",
    image_b64: "dGVzdC1pbWFnZS1iYXNlNjQ=",
    offline_result: {
      referral: "NON_REFERABLE",
      urgency: "ROUTINE",
      grade: 0,
      is_provisional: true,
    },
  });

  // Mock global.fetch to simulate 200 OK from /offline/sync
  const originalFetch = global.fetch;
  global.fetch = async (url: any, opts: any) => {
    return {
      ok: true,
      status: 200,
      json: async () => ({
        screening_id: "server-screening-uuid-001",
        referral: "NON_REFERABLE",
        urgency: "ROUTINE",
        discrepancy_logged: false,
        offline_sync: true,
      }),
    } as any;
  };

  try {
    const summary = await syncQueueWithServer("mock-token", "http://localhost");
    assert.equal(summary.synced, 1);
    assert.equal(summary.failed, 0);

    // Assert item is deleted from IndexedDB after successful sync
    const queuedRemaining = await getQueuedScreenings();
    assert.equal(queuedRemaining.some((q) => q.id === item.id), false);
  } finally {
    global.fetch = originalFetch;
  }
});

test("Failed-sync retry handling: retains item in queue and increments retry_count", async () => {
  const item = await enqueueOfflineScreening({
    patient_pseudonym_id: "PAT-SYNC-FAIL-001",
    eye_side: "OD",
    camera_type: "TABLETOP",
    age: 60,
    diabetes_type: "TYPE_2",
    language: "en",
    image_b64: "dGVzdC1pbWFnZS1iYXNlNjQ=",
    offline_result: {
      referral: "REFERABLE",
      urgency: "PRIORITY",
      grade: 2,
      is_provisional: true,
    },
  });

  // Mock global.fetch to simulate 503 Server Error / network failure
  const originalFetch = global.fetch;
  global.fetch = async () => {
    return {
      ok: false,
      status: 503,
      json: async () => ({ detail: "Server temporarily offline" }),
    } as any;
  };

  try {
    const summary = await syncQueueWithServer("mock-token", "http://localhost");
    assert.equal(summary.failed, 1);
    assert.equal(summary.synced, 0);

    // Verify item remains in queue with incremented retry count
    const queuedRemaining = await getQueuedScreenings();
    const retained = queuedRemaining.find((q) => q.id === item.id);
    assert.ok(retained, "Failed sync item must be retained in queue for retry");
    assert.equal(retained?.status, "failed");
    assert.equal(retained?.retry_count, 1);

    // Clean up
    await removeQueuedScreening(item.id);
  } finally {
    global.fetch = originalFetch;
  }
});
