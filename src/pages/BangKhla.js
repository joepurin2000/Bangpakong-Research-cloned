import React,{useState,useEffect} from 'react'
import MainLayout from '../layouts/MainLayout'
import CurrentForecast from '../components/CurrentForecast'
import HourForecast from '../components/HourForecast'
import Graph from '../components/Graph'
import MapBox from '../components/MapBox'
import DayForecast from '../components/DayForecast'

const textMeter = {
    where: "วัดใหม่บางคล้า",
    address: "ตำบล บางสวน อำเภอ บางคล้า จังหวัด ฉะเชิงเทรา",
    distance: "ระยะห่างจากทะเล 71 กม.",
    link: "https://www.google.com/maps/?q=13.695572,101.164386",
    coordinate: "13°41'44.1\"N 101\°09\'51.8\"E"
  }

const compBangKhlaHourly = "เปรียบเทียบค่าทำนายกับค่าจริงย้อนหลัง: รายชั่วโมง"
const compBangKhlaStyle = {
    line: [{
      name: "กรมชลประทาน",
      type: "monotone",
      dataKey: "gl_actual",
      color: "#1f77b4",
      linename: 'ชลประทาน'
    },
    {
      name: "ทำนาย",
      type: "monotone",
      dataKey: "gl_pred",
      color: "#ff7f0e",
      linename: 'ทำนาย'
    }],
  };

const compBangKhlaDaily = "เปรียบเทียบค่าทำนายกับค่าจริงย้อนหลัง: รายวัน ณ เวลา 6 นาฬิกาตรง"


  const pos = [
    {
        name: "cholpratan-bangkhla",
        longitude: 101.164386,
        latitude: 13.695572,
        color: "cyan"
    }
];


function BangKhla({ socket }) {
    const [data, setData] = useState({ bangkhla: {current:{}, next_24:[], actual_vs_pred:[]} });
    const [daily, setDaily] = useState({ bangkhla: {today: {}, prev: {}, current:{}, next_14:[], actual_vs_pred:[]} });

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
                // console.log(data);
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
            <CurrentForecast data = {data.bangkhla.current} position={textMeter}/>
            {data.bangkhla.next_24.length > 0 && 
            <HourForecast data={data.bangkhla.next_24} length={data.bangkhla.next_24.length} />}
            {data.bangkhla.actual_vs_pred.length > 0 && 
            <Graph data={data.bangkhla.actual_vs_pred} 
            text={compBangKhlaHourly} graph={compBangKhlaStyle} isHourly={true} dataKeyX={"datetime"} />}
            
            <MapBox pos={pos}/>
            
            <DayForecast data={daily.bangkhla}/>
            {daily.bangkhla.actual_vs_pred.length > 0 && 
            <Graph data={daily.bangkhla.actual_vs_pred} 
            text={compBangKhlaDaily} graph={compBangKhlaStyle} isDaily={true} dataKeyX={"date"} />}
        </MainLayout>
    )
}

export default BangKhla