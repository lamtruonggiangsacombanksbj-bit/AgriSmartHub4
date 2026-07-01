import { useEffect } from "react";

const API_ROUTES: Record<string, string> = {
  agrismart_user_accounts: "/api/users",
  agrismart_collaborators_list: "/api/collaborators",
  agrismart_notifs: "/api/notifications",
};

// Simple utility to push local changes to the server
export async function pushToServer(key: string, data: any) {
  const route = API_ROUTES[key];
  if (!route) return;

  try {
    // Write locally first
    localStorage.setItem(key, JSON.stringify(data));

    // Post to backend
    const response = await fetch(route, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      console.error(`Failed to push sync to server for key: ${key}`);
    } else {
      // Dispatch event to notify local components
      window.dispatchEvent(new CustomEvent("agrismart_sync", { detail: { key, data } }));
    }
  } catch (error) {
    console.error(`Error syncing key ${key} to server:`, error);
  }
}

// Background sync function to fetch from server and update local storage if changed
export async function fetchAndSyncFromServer() {
  for (const [key, route] of Object.entries(API_ROUTES)) {
    try {
      const response = await fetch(route);
      if (!response.ok) continue;

      const result = await response.json();
      const serverData = result.data;

      // Check if local storage matches
      const localRaw = localStorage.getItem(key);
      const localData = localRaw ? JSON.parse(localRaw) : null;

      // Deep compare stringified forms
      if (JSON.stringify(serverData) !== JSON.stringify(localData)) {
        localStorage.setItem(key, JSON.stringify(serverData));
        window.dispatchEvent(new CustomEvent("agrismart_sync", { detail: { key, data: serverData } }));
      }
    } catch (e) {
      console.error(`Error syncing from server for key ${key}:`, e);
    }
  }
}

// Custom hook for screens to listen to real-time synchronization updates
export function useSyncListener<T>(key: string, onUpdate: (data: T) => void) {
  useEffect(() => {
    const handleSync = (event: Event) => {
      const customEvent = event as CustomEvent<{ key: string; data: any }>;
      if (customEvent.detail && customEvent.detail.key === key) {
        onUpdate(customEvent.detail.data);
      }
    };

    window.addEventListener("agrismart_sync", handleSync);
    return () => {
      window.removeEventListener("agrismart_sync", handleSync);
    };
  }, [key, onUpdate]);
}
