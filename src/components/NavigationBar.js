import React, { useEffect, useState } from 'react'
import '../styles/NavigationBar.css'
import BlueSaltWater from '../icons/salt-water/03/blue-salt-water.png'
// import YellowSaltWater from '../icons/salt-water/03/yellow-salt-water.png'
// import RedSaltWater from '../icons/salt-water/03/red-salt-water.png'
import LightModeRoundedIcon from '@mui/icons-material/LightModeRounded';
import DarkModeRoundedIcon from '@mui/icons-material/DarkModeRounded';
import MenuRoundedIcon from '@mui/icons-material/MenuRounded';
import CloseRoundedIcon from '@mui/icons-material/CloseRounded';
import HomeRoundedIcon from '@mui/icons-material/HomeRounded';

function NavigationBar(props) {

    const [themeState, setThemeState] = useState(false)

    const switchTheme = () => {
        const newMode = !themeState
        props.onToggleTheme(newMode)
        setThemeState(newMode)
        if (newMode === true) {
            localStorage.setItem('themeMode', 'dark');
        } else if (newMode === false) {
            localStorage.setItem('themeMode', 'light');
        }
    }

    useEffect(() => {
        if (localStorage.getItem('themeMode') === 'dark') {
            setThemeState(true)
        }
    }, []);

    const [clickState, setClickState] = useState(false)
    const handleClick = () => setClickState(!clickState)
    const closeMobileMenu = () => setClickState(false)

    return (
        <nav>
            <div className="nav-con">
                <a className='logo-con' >
                    <figure className='logo-img'>
                        <img src={BlueSaltWater} alt="" />
                    </figure>
                    <div className='logo-text'>
                        <p>Salinity</p>
                        <p>Predictive</p>
                        <p>System</p>

                    </div>
                </a>
                <div className={clickState ? 'menu-con active' : 'menu-con'}>
                    <a className="menu-link" href='/' onClick={closeMobileMenu}>
                        <HomeRoundedIcon className='menu-icon' />
                        <p className='menu-text'>Home</p>
                    </a>
                    <div className="theme-con" onClick={switchTheme}>
                        {themeState ? (
                            <LightModeRoundedIcon className='icon' />
                        ) : (
                            <DarkModeRoundedIcon className='icon' />
                        )}
                        {themeState ? (
                            <p className='theme-text'>Light</p>
                        ) : (
                            <p className='theme-text'>Dark</p>
                        )}
                    </div>
                </div>
                <div className="mobile-menu" onClick={handleClick}>
                    {clickState ? (
                        <CloseRoundedIcon className='icon' />
                    ) : (
                        <MenuRoundedIcon className='icon' />
                    )}
                </div>
            </div>
        </nav>
    )
}

export default NavigationBar