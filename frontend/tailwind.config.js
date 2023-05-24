/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        background: '#623CEA',
        primary: '#F1F2F6',
        secondary: '#271F30',
        danger: '#DD1155'
      }
    }
  },
  plugins: []
}
