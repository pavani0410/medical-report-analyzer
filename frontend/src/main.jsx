    import React from 'react'
    import ReactDOM from 'react-dom/client'
    import App from './App.jsx' // This imports your custom App.jsx
    import './index.css' // This imports the main CSS file

    ReactDOM.createRoot(document.getElementById('root')).render(
      // Removed <React.StrictMode> wrapper
      <App />
    )
    