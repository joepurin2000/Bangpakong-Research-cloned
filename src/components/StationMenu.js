import React from 'react'
import '../styles/StationMenu.css'
import BlueSaltWater from '../icons/salt-water/03/blue-salt-water.png'
// import YellowSaltWater from '../icons/salt-water/03/yellow-salt-water.png'
// import RedSaltWater from '../icons/salt-water/03/red-salt-water.png'
import HomeWorkRoundedIcon from '@mui/icons-material/HomeWorkRounded';
import { mapImg } from './mapImg';

function StationMenu(props) {
    const {link,info,data} = props;
    return (
        <a className='station-menu' href={link}>
            <div className="station-con">
                <div className="title">
                    <HomeWorkRoundedIcon className='icon' />
                    <span className='title-text'>{info.where}</span>
                </div>
                <div className='station-info'>
                    <li>{info.address}</li>
                    <li>{info.distance}</li>
                    <li>{data.date} {data.time} (Indochina Time)</li>
                </div>
            </div>
            <div className="salinity-con">
                <figure className='img-con'>
                    <img src={mapImg(data.gl)} alt="" />
                </figure>
                <div className="salinity-info">
                    <div className="salinity">
                        <div className="salinity-value">
                            <span>{data.gl}</span>
                        </div>
                        <div className="salinity-unit">
                            <span>g/l</span>
                        </div>
                    </div>
                    <div className="ec">
                        <div className="ec-value">
                            <span>{data.uscm}</span>
                        </div>
                        <div className="ec-unit">
                            <span>µS/cm</span>
                        </div>
                    </div>
                </div>
            </div>
        </a>
    )
}

export default StationMenu