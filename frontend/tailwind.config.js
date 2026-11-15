/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        deepteal: "#0B3C49",
        teal: {
          DEFAULT: "#1B7F86",
          50: "#F0F9FA",
          100: "#DDF2F4",
          200: "#B9E4E8",
          600: "#1B7F86",
          700: "#15676D",
          800: "#0F4B50",
          900: "#0B3C49",
        },
        coral: {
          DEFAULT: "#D9622B",
          50: "#FCF3EE",
          100: "#F9E4D9",
          600: "#D9622B",
          700: "#B84E1E",
        },
        pagebg: "#F4F7F8",
      },
    },
  },
  plugins: [],
};


