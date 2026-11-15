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

  function handleScreenerGuest() {
    localStorage.setItem("retinaguard_role", "screener");
    localStorage.setItem("retinaguard_email", "screener@local.phc");
    onLogin("", "screener", "screener@local.phc");
  }

  return (
    <div className="min-h-screen bg-[#F4F7F8] flex flex-col justify-center px-4 py-12 sm:px-6 lg:px-8">
      <div className="w-full max-w-md mx-auto">
        {/* Brand Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-[#E6F3F4] text-[#0B3C49] mb-3 text-2xl font-bold shadow-sm">
            👁
          </div>
          <h1 className="text-3xl font-extrabold text-[#0B3C49] tracking-tight">RetinaGuard</h1>
          <p className="mt-1.5 text-base text-slate-600 font-normal">
            Clinical Decision Support · Diabetic Retinopathy Triage
          </p>
        </div>

        {/* Login Card */}
        <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 sm:p-8">
          <form className="space-y-5" onSubmit={handleSubmit}>
            {error && (
              <div
                role="alert"
                className="rounded-xl bg-[#FCF3EE] p-4 text-sm font-medium text-[#D9622B] border border-[#F9E4D9] flex items-start space-x-2"
              >
                <span className="text-base leading-none">⚠️</span>
                <span>{error}</span>
              </div>
            )}

            <div>
              <label className="block text-sm font-semibold text-slate-800 mb-1.5">
                Email address
              </label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="clinician@hospital.in"
                className="w-full rounded-xl border border-slate-300 px-3.5 py-2.5 text-base text-slate-900 shadow-sm focus:border-[#1B7F86] focus:outline-none focus:ring-2 focus:ring-[#1B7F86]/20 transition"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-slate-800 mb-1.5">
                Password
              </label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full rounded-xl border border-slate-300 px-3.5 py-2.5 text-base text-slate-900 shadow-sm focus:border-[#1B7F86] focus:outline-none focus:ring-2 focus:ring-[#1B7F86]/20 transition"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full flex justify-center items-center py-3 px-4 border border-transparent rounded-xl shadow-sm text-base font-semibold text-white bg-[#1B7F86] hover:bg-[#15676D] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#1B7F86] disabled:opacity-50 transition cursor-pointer"
            >
              {loading ? "Signing in..." : "Sign in"}
            </button>
          </form>

          {allowAnonymous && (
            <div className="mt-5 border-t border-slate-100 pt-4 text-center">
              <button
                type="button"
                onClick={handleScreenerGuest}
                className="text-sm text-[#1B7F86] hover:text-[#0B3C49] font-medium transition cursor-pointer"
              >
                Continue as PHC Screener (Screening only) →
              </button>
            </div>
          )}

          {/* Prototype note under the form */}
          <div className="mt-6 pt-4 border-t border-slate-100 text-center">
            <span className="inline-block px-3 py-1 text-xs font-semibold uppercase tracking-wider text-amber-900 bg-amber-50 rounded-full border border-amber-200">
              Prototype, not clinically validated
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
