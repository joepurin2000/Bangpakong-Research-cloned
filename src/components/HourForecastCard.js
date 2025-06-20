import React from 'react'
import '../styles/HourForecastCard.css'
import {mapImg} from './mapImg'

function HourForecastCard(props) {
    const {data} = props;
    return (
        // <div>HourForecastCard</div>
        <main className="hour-forecast-card">
            <p className='date-text'>{data.date}</p>
            <p className='time-text'>{data.time}</p>
            <figure className="img-con">
                {/* <img src={BlueSaltWater} alt="" /> */}
                <img src={mapImg(data.gl)} alt="" />
            </figure>
            <div className="salinity-info">
                <p className='salinity-value'>{data.gl}</p>
                <p className='salinity-unit'>g/l</p>
            </div>
            <div className="ec-info">
                <p className='ec-value'>{data.uscm}</p>
                <p className='ec-unit'>µS/cm</p>
            </div>
        </main>
    )
}

export default HourForecastCard