import React from 'react'
import '../styles/DaysItem.css'
import BlueSaltWater from '../icons/salt-water/03/blue-salt-water.png'
// import YellowSaltWater from '../icons/salt-water/03/yellow-salt-water.png'
// import RedSaltWater from '../icons/salt-water/03/red-salt-water.png'
import ElectricBoltRoundedIcon from '@mui/icons-material/ElectricBoltRounded';
import { mapImg } from './mapImg';

function DaysItem(props) {
    const {isToday = false, data} = props;
    return (
        <div className="days-item">
            <div className="day-text">
            {isToday ? "วันนี้" : data.date}
            </div>
            <div className="info-con">
                <div className="salinity">
                    <div className="img-con">
                        <img src={mapImg(data.gl)} alt="" />
                    </div>
                    <div className="salinity-info">
                        <div className="salinity-value">
                            {data.gl}
                        </div>
                        <div className="salinity-unit">
                            g/l
                        </div>
                    </div>
                </div>
                <div className="ec">
                    <ElectricBoltRoundedIcon className='ec-icon' />
                    <div className="ec-info">
                        <div className="ec-value">
                            
                            {data.uscm}
                        </div>
                        <div className="ec-unit">
                            µS/cm
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default DaysItem