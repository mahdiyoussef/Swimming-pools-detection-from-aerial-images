/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                pool: {
                    light: '#4fd1c5',
                    DEFAULT: '#38b2ac',
                    dark: '#2c7a7b',
                }
            }
        },
    },
    plugins: [],
}
