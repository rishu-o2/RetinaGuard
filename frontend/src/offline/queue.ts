export interface OfflineScreeningItem {
  id: string;
  patient_pseudonym_id: string;
  eye_side: "OD" | "OS" | string;
  camera_type: string;
  age: number;
  diabetes_type: string;
  diabetes_duration_years?: number | null;
  hba1c?: number | null;
  is_pregnant: boolean;
  language: string;
  image_b64: string;
  offline_result: Record<string, any>;
  created_at: string;
  status: "pending" | "syncing" | "failed";
  retry_count: number;
}

export interface SyncResultSummary {
  synced: number;
  failed: number;
  discrepancies: number;
  items: Array<{ id: string; server_result?: any; error?: string }>;
}

const DB_NAME = "RetinaGuardOfflineDB";
const STORE_NAME = "screening_queue";
const DB_VERSION = 1;

/** Open or initialize the IndexedDB queue database */
export function openQueueDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("IndexedDB is not available in this environment."));
      return;
    }

    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (e: IDBVersionChangeEvent) => {
      const db = (e.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/** Enqueue an offline screening record */
export async function enqueueOfflineScreening(
  item: Omit<OfflineScreeningItem, "id" | "created_at" | "status" | "retry_count"> & {
    id?: string;
  }
): Promise<OfflineScreeningItem> {
  const db = await openQueueDB();
  const queueItem: OfflineScreeningItem = {
    ...item,
    id: item.id || `offline-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    created_at: new Date().toISOString(),
    status: "pending",
    retry_count: 0,
  };

  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const req = store.put(queueItem);

    req.onsuccess = () => resolve(queueItem);
    req.onerror = () => reject(req.error);
  });
}

/** Retrieve all queued screening items */
export async function getQueuedScreenings(): Promise<OfflineScreeningItem[]> {
  const db = await openQueueDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const req = store.getAll();

    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(req.error);
  });
}

/** Delete a specific queued item after confirmed synchronization */
export async function removeQueuedScreening(id: string): Promise<void> {
  const db = await openQueueDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const req = store.delete(id);

    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

/** Update status and retry count for a failed item */
export async function updateQueuedItemStatus(
  id: string,
  status: "pending" | "failed"
): Promise<void> {
  const db = await openQueueDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const getReq = store.get(id);

    getReq.onsuccess = () => {
      const item: OfflineScreeningItem = getReq.result;
      if (item) {
        item.status = status;
        item.retry_count = (item.retry_count || 0) + 1;
        store.put(item);
      }
      resolve();
    };
    getReq.onerror = () => reject(getReq.error);
  });
}

/** Synchronize all queued items with the backend /offline/sync endpoint */
export async function syncQueueWithServer(
  token?: string | null,
  apiBaseUrl: string = ""
): Promise<SyncResultSummary> {
  const items = await getQueuedScreenings();
  const summary: SyncResultSummary = {
    synced: 0,
    failed: 0,
    discrepancies: 0,
    items: [],
  };

  if (items.length === 0) {
    return summary;
  }

  for (const item of items) {
    try {
      const resp = await fetch(`${apiBaseUrl}/offline/sync`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          patient_pseudonym_id: item.patient_pseudonym_id,
          eye_side: item.eye_side,
          camera_type: item.camera_type,
          age: item.age,
          diabetes_type: item.diabetes_type,
          diabetes_duration_years: item.diabetes_duration_years,
          hba1c: item.hba1c,
          is_pregnant: item.is_pregnant,
          language: item.language,
          image_b64: item.image_b64,
          offline_result: item.offline_result,
        }),
      });

      if (!resp.ok) {
        throw new Error(`Sync HTTP error ${resp.status}`);
      }

      const serverResult = await resp.json();

      // Confirmed 200 OK: remove item from queue
      await removeQueuedScreening(item.id);
      summary.synced++;
      if (serverResult.discrepancy_logged) {
        summary.discrepancies++;
      }
      summary.items.push({ id: item.id, server_result: serverResult });
    } catch (err: any) {
      // Sync failed: retain item in IndexedDB and increment retry count
      await updateQueuedItemStatus(item.id, "failed");
      summary.failed++;
      summary.items.push({ id: item.id, error: err.message || "Network error" });
    }
  }

  return summary;
}
