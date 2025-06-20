import React, { createContext, useEffect, useState } from 'react'
import './App.css';
import NavigationBar from './components/NavigationBar';
import CurrentForecast from './components/CurrentForecast';
import HourForecast from './components/HourForecast';
import HourForecastCard from './components/HourForecastCard';
import Footer from './components/Footer';
import MapBox from './components/MapBox';
import socketIOClient from "socket.io-client";
import Graph from "./components/Graph";

export const ThemeContext = createContext(null)


const textBangklaMeter = "เปรียบเทียบค่าความเค็ม (g/l) ระหว่าง เครื่องวัด กับ กรมชลประทาน 24 ชม. ที่ผ่านมา";
const styleBangklaMeter = {
  line: [{
    name: "กรมชลประทาน",
    type: "monotone",
    dataKey: "bangkla",
    color: "#4DACFF",
    linename: 'ชลประทาน'
  },
  {
    name: "meter",
    type: "monotone",
    dataKey: "meter",
    color: "#82ca9d",
    linename: 'วัดคลองเขื่อน'
  }],
  refLineY: [
    {
      y: 2,
      color: "red",
      linestyle: "5 5"
    },
    {
      y: .5,
      color: "green",
      linestyle: "5 5"
    }
  ]
};

const textMeter = {
  where: "หน้าวัดคลองเขื่อน",
  address: "ตำบล คลองเขื่อน อำเภอ คลองเขื่อน ฉะเชิงเทรา",
  distance: "ระยะห่างจากทะเล 86 กม.",
  link: "https://goo.gl/maps/rhut5YAiXzS17DCw6",
  coordinate: "13°45'39.3\"N, 101\°11\'36.0\"E"
}


function App() {

  const [theme, setTheme] = useState('light');

  const [data, setData] = useState({ current: {}, next_24: [], comp_chol_meter: [] });

  const socket = socketIOClient("http://kmitl.duckdns.org:20001/hourly");


  useEffect(() => {
    fetch("http://kmitl.duckdns.org:20001/", {
      headers: {
        "Content-Type": "application/json",
      }
    })
      .then(response => response.json())
      .then(data => {
        // console.log(data);
        setData(data);
      })
  }, []);


  useEffect(() => {
    socket.on("post_hourly", (data) => {
      const parsedData = JSON.parse(data);
      // console.log(parsedData.next_24);
      setData(parsedData);
    });
    return () => {
      socket.off("post_hourly");
    };
  });


  const toggleTheme = (state) => {
    if (state === true) {
      setTheme('dark')
    } else if (state === false) {
      setTheme('light')
    }
  }

  return (
    // <div>App</div>
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      <div className='App' id={theme}>
        <NavigationBar onToggleTheme={toggleTheme} />
        <div className="card-con">
          <div className="card-list">

            {/* วัดคลองเขื่อน */}
            <CurrentForecast data={data.current} position={textMeter} />

            {/* ชลประทาน บางคล้า */}
            {data.next_24.length > 0 && <HourForecast data={data.next_24} length={data.next_24.length} />}

            {/* บางคล้า vs วัดคลองเขื่อน */}
            {/* {data.comp_chol_meter.length > 0 && <Graph data={data.comp_chol_meter} text={textBangKhla} graph={styleBangklaMeter} />} */}

            {/* แผนที่ (ตอนนี้แสดงรวมทุกจุด) */}
            <MapBox />
          </div>
        </div>
        <Footer />
      </div>
    </ThemeContext.Provider>
  )
}

export default App
