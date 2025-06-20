import React from 'react'
import '../styles/DayForecast.css'
import TodayRoundedIcon from '@mui/icons-material/TodayRounded';
import DaysItem from './DaysItem';
import YesterdayItem from './YesterdayItem';

function DayForecast(props) {
    const {data} = props;

    const cardElements = Array.from({ length: data.next_14.length }, (_, i) => (
        <DaysItem key={i} data={data.next_14[i]}/>
      ));

    return (
        <div className="day-forecast">
            <YesterdayItem data={data.prev}/>

            <DaysItem isToday={true} data={data.current}/>

            {cardElements}
            
            <div className="footer">
                <TodayRoundedIcon className='today-icon' />
                <div className="today-con">
                    <div className="today-text">
                        วันนี้:
                    </div>
                    <div className="date-text">
                        {data.today.date}
                    </div>
                </div>
            </div>
        </div>
    )
}

export default DayForecast