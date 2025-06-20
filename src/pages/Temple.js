import React,{useState,useEffect} from 'react'
import MainLayout from '../layouts/MainLayout'
import CurrentForecast from '../components/CurrentForecast'
import HourForecast from '../components/HourForecast'
import Graph from '../components/Graph'
import MapBox from '../components/MapBox'
import DayForecast from '../components/DayForecast'

const textMeter = {
    where: "หน้าวัดคลองเขื่อน",
    address: "ตำบล คลองเขื่อน อำเภอ คลองเขื่อน ฉะเชิงเทรา",
    distance: "ระยะห่างจากทะเล 86 กม.",
    link: "https://goo.gl/maps/rhut5YAiXzS17DCw6",
    coordinate: "13°45'39.3\"N, 101\°11\'36.0\"E"
  }

  const compTempleHourly = "เปรียบเทียบค่าทำนายกับค่าจริงย้อนหลัง: รายชั่วโมง"
  const compTempleDaily = "เปรียบเทียบค่าทำนายกับค่าจริงย้อนหลัง: รายวัน ณ เวลา 6 นาฬิกาตรง"
  const compTempleStyle = {
      line: [{
        name: "เครื่องวัด",
        type: "monotone",
        dataKey: "gl_actual",
        color: "#1f77b4",
        linename: 'วัดคลองเขื่อน'
      },
      {
        name: "ทำนาย",
        type: "monotone",
        dataKey: "gl_pred",
        color: "#ff7f0e",
        linename: 'ทำนาย'
      }],
      refLineY: [
      ]
    };

const pos = [
    {
        name: "meter",
        longitude: 101.193333,
        latitude: 13.760917,
        color: "red"
    }
];

function Temple({ socket }) {
  const [data, setData] = useState({ temple: {current:{}, next_24:[], actual_vs_pred:[]} });
  const [daily, setDaily] = useState({ temple: {today: {}, prev: {}, current:{}, next_14:[], actual_vs_pred: []} });

  useEffect(() => {
      fetch("http://kmitl.duckdns.org:20001/", {
          headers: {
              "Content-Type": "application/json",
          }
      })
          .then(response => response.json())
          .then(data => {
              // console.log(data.next_24);
              setData(data);
          })
  }, []);

  useEffect(() => {
      fetch("http://kmitl.duckdns.org:20001/daily", {
          headers: {
              "Content-Type": "application/json",
          }
      })
          .then(response => response.json())
          .then(data => {
              // console.log(data.next_24);
              setDaily(data);
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

  useEffect(() => {
      socket.on("post_daily", (data) => {
          const parsedData = JSON.parse(data);
          setDaily(parsedData);
      });
      return () => {
          socket.off("post_daily");
      };
  });

  return (
      <MainLayout>
          <CurrentForecast data = {data.temple.current} position={textMeter}/>
          {data.temple.next_24.length > 0 && <HourForecast data={data.temple.next_24} length={data.temple.next_24.length} />}
          {data.temple.actual_vs_pred.length > 0 && <Graph data={data.temple.actual_vs_pred} text={compTempleHourly} graph={compTempleStyle} isHourly={true} dataKeyX={"datetime"}/>}
          <MapBox pos={pos}/>
          <DayForecast data={daily.temple}/>
          {daily.temple.actual_vs_pred.length > 0 && <Graph data={daily.temple.actual_vs_pred} text={compTempleDaily} graph={compTempleStyle} isDaily={true} dataKeyX={"date"}/>}
      </MainLayout>
  )
}

export default Temple;