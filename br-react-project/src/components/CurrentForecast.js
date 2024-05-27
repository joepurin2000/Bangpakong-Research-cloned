import React, { Component } from 'react'
import '../styles/CurrentForecast.css'
import HomeWorkRoundedIcon from '@mui/icons-material/HomeWorkRounded';
import LocationOnRoundedIcon from '@mui/icons-material/LocationOnRounded';
import WaterRoundedIcon from '@mui/icons-material/WaterRounded';
import MapRoundedIcon from '@mui/icons-material/MapRounded';
import OpacityRoundedIcon from '@mui/icons-material/OpacityRounded';
import ElectricBoltRoundedIcon from '@mui/icons-material/ElectricBoltRounded';
import WatchLaterRoundedIcon from '@mui/icons-material/WatchLaterRounded';
import {mapImg} from './mapImg'


function CurrentForecast(props) {
    const {data,position} = props; //same as -> const data = props.data;
    // console.log("Data in CurrentForecast:", data);
    return (
        // <div>CurrentForecast</div>
        <main className="current-forecast">
            
            <div className="header">
                <div className="title-1">
                    <HomeWorkRoundedIcon className='icon' sx={{ fontSize: 24 }} />
                    <p className='text-1'>{position.where}</p>
                </div>
            </div>
            <div className="header">
                <div className="title-2">
                    <LocationOnRoundedIcon className='icon' sx={{ fontSize: 24 }} />
                    <p className='text-2'>{position.address}</p>
                </div>
                <div className="title-2">
                    <WaterRoundedIcon className='icon' sx={{ fontSize: 24 }} />
                    <p className='text-2'>{position.distance}</p>
                </div>
                <div className="title-2">
                    <MapRoundedIcon className='icon' sx={{ fontSize: 24 }} />
                    <a className='link-text' href={position.link} target='_blank'>{position.coordinate}</a>
                </div>
            </div>
            <div className="info">
                <div className="info-con">
                    <div className="salinity-con">
                        <div className="title-1">
                            <OpacityRoundedIcon className='icon' sx={{ fontSize: 24 }} />
                            <p className='text-2'>ความเค็ม</p>
                        </div>
                        <div className="salinity-info">
                            {/* <p className='salinity-value'>99.99</p> */}
                            <p className='salinity-value'>{data.gl}</p>
                            <p className='salinity-unit'>g/l</p>
                        </div>
                    </div>
                    <div className="ec-con">
                        <div className="title-1">
                            <ElectricBoltRoundedIcon className='icon' sx={{ fontSize: 24 }} />
                            <p className='text-2'>ค่าการนำไฟฟ้า</p>
                        </div>
                        <div className="ec-info">
                            {/* <p className='ec-value'>99999.99</p> */}
                            <p className='ec-value'>{data.uscm}</p>
                            <p className='ec-unit'>µS/cm</p>
                        </div>
                    </div>
                </div>

                <figure className="img-con">
                    {/* <img src={BlueSaltWater} alt="" /> */}
                    <img src={mapImg(data.gl)}/>

                </figure>
            </div>
            <div className="footer">
                <div className="title-1">
                    <WatchLaterRoundedIcon className='icon' sx={{ fontSize: 24 }} />
                    {/* <p className='text-1'>ศ. 00/00/0000 00:00 GMT+07:00</p> */}
                    <p className='text-1'>{data.date} {data.time} GMT+07:00</p>
                </div>
            </div>
        </main>
    )
}

export default CurrentForecast