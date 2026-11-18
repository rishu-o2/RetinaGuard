import { useState } from "react";
import { Mail, Lock, Eye, EyeOff } from "lucide-react";

interface LoginPageProps {
  onLogin: (token: string, role: string, email: string) => void;
  allowAnonymous?: boolean;
}

export function LoginPage({ onLogin, allowAnonymous = false }: LoginPageProps) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (!email.trim() || !password.trim()) {
      setError("Please enter both email and password.");
      return;
    }

    setIsSubmitting(true);
    try {
      const params = new URLSearchParams();
      params.append("username", email.trim());
      params.append("password", password);

      const res = await fetch("/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: params.toString(),
      });

      if (!res.ok) {
        if (res.status === 401) {
          setError("Incorrect email or password");
        } else if (res.status === 429) {
          setError("Too many login attempts. Please wait a minute.");
        } else {
          setError("Login failed. Please check credentials or contact administrator.");
        }
        return;
      }

      const data = await res.json();
      onLogin(data.access_token, data.role || "screener", email.trim());
    } catch {
      setError("Unable to connect to the authentication service. Check server status.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleGuestScreener = () => {
    onLogin("", "screener", "screener@local.phc");
  };

  return (
    <main className="relative min-h-[100dvh] w-full flex items-center justify-center p-4 sm:p-6 overflow-hidden bg-[#020612] font-sans">
      {/* Background Radial Ambient Glows */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full bg-teal-500/10 blur-3xl pointer-events-none select-none" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 rounded-full bg-emerald-500/10 blur-3xl pointer-events-none select-none" />

      {/* AgriNexus Style Two-Column Elevated Card */}
      <div className="relative z-10 w-full max-w-[880px] rounded-[28px] sm:rounded-[34px] bg-[#050b1a] border border-slate-800/90 shadow-[0_25px_70px_-15px_rgba(0,0,0,0.7)] overflow-hidden flex flex-col md:flex-row">
        
        {/* LEFT COLUMN: Pure CSS/SVG Branded Ocular Sphere Visual */}
        <div className="w-full md:w-[48%] bg-gradient-to-br from-[#061626] via-[#092238] to-[#04101c] p-8 sm:p-10 flex flex-col justify-between border-b md:border-b-0 md:border-r border-slate-800/80 relative overflow-hidden">
          {/* Subtle grid background */}
          <div
            className="absolute inset-0 opacity-20 pointer-events-none"
            style={{
              backgroundImage:
                "linear-gradient(to right, rgba(20,184,166,0.15) 1px, transparent 1px), linear-gradient(to bottom, rgba(20,184,166,0.15) 1px, transparent 1px)",
              backgroundSize: "20px 20px",
            }}
          />

          {/* Top Wordmark & Badges */}
          <div className="relative z-10">
            <div className="flex items-center gap-2 mb-2">
              <span className="w-2.5 h-2.5 rounded-full bg-teal-400 animate-pulse shadow-[0_0_8px_#2dd4bf]" />
              <span className="text-[11px] font-mono uppercase tracking-widest text-teal-300 font-bold">
                Clinical AI Platform
              </span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              RetinaGuard
            </h1>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Autonomous Retinal Screening Decision Support
            </p>
          </div>

          {/* Glowing Ocular / Retinal Visual (Pure CSS & SVG) */}
          <div className="relative my-8 py-4 flex items-center justify-center z-10">
            <div className="relative w-48 h-48 sm:w-56 sm:h-56 flex items-center justify-center">
              {/* Outer Pulsing Rings */}
              <div className="absolute inset-0 rounded-full border border-teal-500/20 animate-ping opacity-30" />
              <div className="absolute -inset-3 rounded-full border border-cyan-400/20 animate-pulse" />

              {/* Glowing Orb Sphere */}
              <div className="w-40 h-40 sm:w-48 sm:h-48 rounded-full bg-gradient-to-tr from-[#0b3c49] via-[#0e4e5e] to-[#14b8a6] p-1 shadow-[0_0_50px_rgba(20,184,166,0.35),inset_0_0_30px_rgba(6,182,212,0.4)] flex items-center justify-center relative">
                {/* Ocular Fundus Vascular Tree SVG */}
                <svg viewBox="0 0 200 200" fill="none" className="w-full h-full opacity-80">
                  <defs>
                    <radialGradient id="retinaGlow" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="#2dd4bf" stopOpacity="0.8" />
                      <stop offset="55%" stopColor="#0f766e" stopOpacity="0.4" />
                      <stop offset="100%" stopColor="#042f2e" stopOpacity="0.1" />
                    </radialGradient>
                  </defs>
                  {/* Fundus glow base */}
                  <circle cx="100" cy="100" r="92" fill="url(#retinaGlow)" />
                  {/* Optic Disc */}
                  <circle cx="68" cy="100" r="16" fill="#fef08a" opacity="0.85" filter="drop-shadow(0 0 6px #facc15)" />
                  <circle cx="68" cy="100" r="8" fill="#ffffff" opacity="0.9" />
                  {/* Fovea / Macula */}
                  <circle cx="128" cy="102" r="14" fill="#042f2e" opacity="0.6" stroke="#14b8a6" strokeWidth="1" strokeDasharray="2 2" />
                  <circle cx="128" cy="102" r="3" fill="#2dd4bf" opacity="0.9" />
                  {/* Retinal Blood Vessels (Superior & Inferior Arcades) */}
                  <path d="M68 94 C75 70, 95 50, 130 45 C150 42, 175 55, 185 65" stroke="#ef4444" strokeWidth="2.4" strokeLinecap="round" opacity="0.75" />
                  <path d="M68 106 C78 130, 98 152, 134 156 C155 158, 176 148, 186 138" stroke="#ef4444" strokeWidth="2.4" strokeLinecap="round" opacity="0.75" />
                  <path d="M68 96 C85 85, 110 80, 145 84" stroke="#f87171" strokeWidth="1.6" strokeLinecap="round" opacity="0.8" />
                  <path d="M68 104 C85 115, 110 118, 142 115" stroke="#f87171" strokeWidth="1.6" strokeLinecap="round" opacity="0.8" />
                  {/* Secondary Capillaries */}
                  <path d="M110 58 Q125 70 140 68" stroke="#fca5a5" strokeWidth="1.1" strokeLinecap="round" opacity="0.7" />
                  <path d="M114 142 Q130 132 148 136" stroke="#fca5a5" strokeWidth="1.1" strokeLinecap="round" opacity="0.7" />
                  <path d="M60 98 Q45 92 30 94" stroke="#ef4444" strokeWidth="1.4" strokeLinecap="round" opacity="0.6" />
                </svg>

                {/* Central AI Targeting Crosshair */}
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <div className="w-16 h-16 rounded-full border border-teal-300/40 flex items-center justify-center">
                    <div className="w-2 h-2 rounded-full bg-teal-300 animate-ping" />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Protocol Info */}
          <div className="relative z-10 pt-2 border-t border-slate-800/80">
            <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
              <span>5-Agent Orchestration</span>
              <span className="text-teal-400">Vision • Safety • Audit</span>
            </div>
            <p className="text-[10px] text-slate-500 mt-1 uppercase tracking-wider">
              Prototype, not clinically validated
            </p>
          </div>
        </div>

        {/* RIGHT COLUMN: AgriNexus Welcome & Form */}
        <div className="w-full md:w-[52%] p-8 sm:p-11 flex flex-col justify-center bg-[#071120]">
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-white tracking-tight">Welcome Back</h2>
            <p className="text-xs sm:text-sm text-slate-400 mt-1">
              Please enter your details to continue.
            </p>
          </div>

          {/* Inline Error Alert */}
          {error && (
            <div
              role="alert"
              className="mb-5 p-3 rounded-xl bg-red-950/60 border border-red-500/50 text-red-200 text-xs flex items-center gap-2 animate-in fade-in"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-red-400 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Email Field */}
            <div>
              <label htmlFor="login-email" className="block text-xs font-semibold text-slate-300 mb-1.5">
                Email Address
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                  <Mail className="w-4 h-4" />
                </div>
                <input
                  id="login-email"
                  type="email"
                  autoComplete="username"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="admin@hospital.in"
                  className="w-full pl-10 pr-3.5 py-2.5 rounded-xl bg-slate-900/90 border border-slate-700 text-white text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent transition"
                />
              </div>
            </div>

            {/* Password Field */}
            <div>
              <label htmlFor="login-password" className="block text-xs font-semibold text-slate-300 mb-1.5">
                Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                  <Lock className="w-4 h-4" />
                </div>
                <input
                  id="login-password"
                  type={showPassword ? "text" : "password"}
                  autoComplete="current-password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  className="w-full pl-10 pr-10 py-2.5 rounded-xl bg-slate-900/90 border border-slate-700 text-white text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent transition"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 pr-3.5 flex items-center text-slate-400 hover:text-slate-200 focus:outline-none cursor-pointer"
                  title={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* Remember Me Checkbox */}
            <div className="flex items-center justify-between pt-1">
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                  className="w-4 h-4 rounded border-slate-700 bg-slate-900 text-teal-600 focus:ring-teal-500 cursor-pointer"
                />
                <span className="text-xs text-slate-400 font-medium">Remember me</span>
              </label>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full mt-2 py-3 px-4 rounded-xl font-bold text-sm bg-gradient-to-r from-teal-500 via-teal-600 to-emerald-600 text-slate-950 hover:brightness-110 active:scale-[0.99] transition shadow-lg shadow-teal-900/30 cursor-pointer disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {isSubmitting ? (
                <>
                  <div className="w-4 h-4 rounded-full border-2 border-slate-950 border-t-transparent animate-spin" />
                  <span>Please wait...</span>
                </>
              ) : (
                <>
                  <span>Login</span>
                  <span className="sr-only">Sign in</span>
                </>
              )}
            </button>
          </form>

          {/* Admin Created Note */}
          <div className="mt-5 pt-4 border-t border-slate-800/80 text-center">
            <p className="text-[11px] text-slate-400 leading-normal">
              Accounts are created by your administrator.
            </p>
          </div>

          {/* Anonymous PHC Screener Option (rendered for guest testing) */}
          {allowAnonymous && (
            <div className="mt-3 text-center">
              <button
                type="button"
                onClick={handleGuestScreener}
                className="text-xs text-teal-400 hover:text-teal-300 underline font-semibold transition cursor-pointer"
              >
                Continue as PHC Screener
              </button>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

export default LoginPage;
