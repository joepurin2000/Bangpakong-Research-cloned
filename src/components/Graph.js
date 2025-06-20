import React, { Component, useState} from "react";
import "../styles/Graph.css"
import SimpleLineChart from "./SimpleLineChart";

import ShowChartIcon from '@mui/icons-material/ShowChart';
import HistoryRoundedIcon from '@mui/icons-material/HistoryRounded';


function Graph(props) {
    const {isHourly = false, isDaily = false} = props;
    const [timeRange, setTimeRange] = useState(isHourly ? 24 : isDaily ? 30 : props.data.length);

    const handleTimeRangeChange = (event) => {
    setTimeRange(event.target.value);
    }

    return (
        <main className="graph-box">
            <div className="header">
                <div className="title-1">
                    <ShowChartIcon className='icon' sx={{ fontSize: 24 }} />
                    <p className='text-1'>{props.text}</p>
                </div>
            </div>
            <div className="select-con">
                <HistoryRoundedIcon className='icon' />
                <form>
                    <select name="time-range" id="time-range" className='select-box' value={timeRange} onChange={handleTimeRangeChange}>
                        {isHourly && <option value={24} className='select-text'>1 วัน</option>}
                        {isHourly && <option value={168} className='select-text'>1 สัปดาห์</option>}
                        
                        {isDaily && <option value={30} className='select-text'>1 เดือน</option>}
                        {isDaily && <option value={90} className='select-text'>3 เดือน</option>}

                        <option value={props.data.length} className='select-text'>ทั้งหมด</option>
                    </select>
                </form>
            </div>
            <div className="graph-con">
                <SimpleLineChart data={props.data} graph={props.graph} timeRange={timeRange} dataKeyX={props.dataKeyX}/>
            </div>
        </main>
    )

}

export default Graph;