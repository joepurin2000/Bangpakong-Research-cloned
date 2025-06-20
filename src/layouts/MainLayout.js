import React, { createContext, useEffect, useState } from 'react'
import '../styles/MainLayout.css'
import NavigationBar from '../components/NavigationBar';
import Footer from '../components/Footer';

export const ThemeContext = createContext(null)

function MainLayout({ children }) {
    
    const [theme, setTheme] = useState('light');

    const toggleTheme = (state) => {
        if (state === true) {
            setTheme('dark')
        } else if (state === false) {
            setTheme('light')
        }
    }

    useEffect(() => {
        if (localStorage.getItem('themeMode') === 'dark') {
            const isDarkMode = localStorage.getItem('themeMode')
            setTheme(isDarkMode);
        }
    }, []);

    return (
        <ThemeContext.Provider value={{ theme, toggleTheme }}>
            <div id={theme}>
                <NavigationBar onToggleTheme={toggleTheme} />
                <div className="card-con">
                    <div className="card-list">

                        {children}

                    </div>
                </div>
                <Footer />
            </div>
        </ThemeContext.Provider>
    )
}

export default MainLayout