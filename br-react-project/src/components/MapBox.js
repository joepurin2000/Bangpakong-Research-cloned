import React from 'react'
import Map, {
    Marker,
    NavigationControl,
    FullscreenControl,
    GeolocateControl,
} from "react-map-gl";
import '../styles/MapBox.css';
import 'mapbox-gl/dist/mapbox-gl.css';
import LocationOnRoundedIcon from '@mui/icons-material/LocationOnRounded';


function MapBox(props) {
    const {pos} = props;
    const markers = pos.map((marker, index) => (
        <Marker
            key={index}
            longitude={marker.longitude}
            latitude={marker.latitude}
            color={marker.color}
        />
    ));

    return (
        <main className="map-box">
            <div className="header">
                <div className="title-1">
                    <LocationOnRoundedIcon className='icon' sx={{ fontSize: 24 }} />
                    <p className='text-1'>ตำแหน่งสถานี</p>
                </div>
            </div>
            <div className="map-con">
                <Map
                    mapboxAccessToken="pk.eyJ1Ijoiam9lY2VrbWl0bCIsImEiOiJjbGRsdmVpMGEwMmk1NDFuMGFpdnV6NWl2In0.oyKImg2FLZ0ZemRrSWDhgA"
                    initialViewState={{
                        longitude: pos[0].longitude,
                        latitude: pos[0].latitude,
                        zoom: 10.5
                    }}
                    style={{
                        width: '100%',
                        height: '100%',
                        borderRadius: '8px',
                    }}
                    mapStyle="mapbox://styles/mapbox/streets-v12"
                >

                    {markers}
                    <NavigationControl position='bottom-right' />
                    <FullscreenControl position='bottom-right' />
                    <GeolocateControl position='bottom-right' />
                </Map>
                {/* <div className="rectangle"></div> */}
            </div>
        </main>
    )
}

export default MapBox