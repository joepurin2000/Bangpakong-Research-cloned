import React from 'react'
import '../styles/HourForecast.css'
import OpacityRoundedIcon from '@mui/icons-material/OpacityRounded';
import HourForecastCard from './HourForecastCard';

function HourForecast(props) {
    const {data, length} = props;

    // console.log("HourForecast data:", data);

    const cardElements = Array.from({ length: length }, (_, i) => (
        <HourForecastCard key={i} data={data[i]}/>
      ));

    return (
        // <div>HourForecast</div>
        <main className="hour-forecast">
            <div className="header">
                <div className="title-1">
                    <OpacityRoundedIcon className='icon' sx={{ fontSize: 24 }} />
                    <p className='text-1'>พยากรณ์ความเค็มล่วงหน้า {length} ชั่วโมง</p>
                </div>
            </div>
            <div className="item-list">
                <div className="item-con">
                    {cardElements}
                </div>
            </div>
        </main>
    )
}

export default HourForecast