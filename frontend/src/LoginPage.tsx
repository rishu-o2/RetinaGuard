import React, { useState } from "react";

interface LoginPageProps {
  onLogin: (token: string, role: string, email: string) => void;
  allowAnonymous?: boolean;
}

export function LoginPage({ onLogin, allowAnonymous: propAllowAnonymous }: LoginPageProps) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [allowAnonymous, setAllowAnonymous] = useState<boolean>(propAllowAnonymous ?? false);

  React.useEffect(() => {
    if (propAllowAnonymous !== undefined) return;
    fetch("/auth/config")
      .then((res) => res.json())
      .then((data) => {
        if (data && typeof data.allow_anonymous_screening === "boolean") {
          setAllowAnonymous(data.allow_anonymous_screening);
        }
      })
      .catch(() => {});
  }, [propAllowAnonymous]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const formData = new URLSearchParams();
      formData.append("username", email);
      formData.append("password", password);

      const res = await fetch("/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: formData.toString(),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Authentication failed. Please check credentials.");
      }

      const data = await res.json();
      localStorage.setItem("retinaguard_token", data.access_token);
      localStorage.setItem("retinaguard_role", data.role);
      localStorage.setItem("retinaguard_email", data.email);
      onLogin(data.access_token, data.role, data.email);
    } catch (err: any) {
      setError(err.message || "Failed to log in");
    } finally {
      setLoading(false);
    }
  }

  // Quick screener guest mode for quick access in clinic
  function handleScreenerGuest() {
    localStorage.setItem("retinaguard_role", "screener");
    localStorage.setItem("retinaguard_email", "screener@local.phc");
    onLogin("", "screener", "screener@local.phc");
  }

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md text-center">
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">RetinaGuard</h1>
        <p className="mt-1 text-sm text-slate-600">
          Clinical Decision-Support Tool · Diabetic Retinopathy Triage
        </p>
        <div className="mt-3 inline-block rounded border-2 border-amber-500 bg-amber-50 px-3 py-1 text-xs font-bold text-amber-900 tracking-wide uppercase">
          ⚠ Prototype, not clinically validated
        </div>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10 border border-slate-200">
          <form className="space-y-5" onSubmit={handleSubmit}>
            {error && (
              <div className="rounded-md bg-red-50 p-3 text-sm text-red-700 border border-red-200">
                {error}
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-slate-700">Email address</label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="clinician@hospital.in"
                className="mt-1 block w-full rounded-md border border-slate-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 sm:text-sm"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700">Password</label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="mt-1 block w-full rounded-md border border-slate-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 sm:text-sm"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full flex justify-center py-2.5 px-4 border border-transparent rounded-md shadow-sm text-sm font-semibold text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
            >
              {loading ? "Signing in..." : "Sign in"}
            </button>
          </form>

            {allowAnonymous && (
              <div className="mt-6 border-t border-slate-200 pt-4 text-center">
                <button
                  type="button"
                  onClick={handleScreenerGuest}
                  className="text-xs text-slate-500 hover:text-indigo-600 font-medium"
                >
                  Continue as PHC Screener (Screening only) →
                </button>
              </div>
            )}
        </div>
      </div>
    </div>
  );
}

