/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{html,ts}",
  ],
  theme: {
    extend: {
      colors: {
        text: '#1A1A1A',
        background: '#F5F7FA',
        primary: '#2A9D8F',
        secondary: '#264653',
        accent: '#E76F51',
      }
    },
  },
  plugins: [],
}

