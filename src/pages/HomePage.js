import React,{useState,useEffect} from 'react'
import MainLayout from '../layouts/MainLayout'
import TitleBanner from '../components/TitleBanner'
import StationMenu from '../components/StationMenu'

const textMeter = {
    where: "หน้าวัดคลองเขื่อน",
    address: "ตำบล คลองเขื่อน อำเภอ คลองเขื่อน ฉะเชิงเทรา",
    distance: "ระยะห่างจากทะเล 86 กม.",
  }

const textBangkhla = {
    where: "วัดใหม่บางคล้า",
    address: "ตำบล บางสวน อำเภอ บางคล้า จังหวัด ฉะเชิงเทรา",
    distance: "ระยะห่างจากทะเล 71 กม.",
}
  

function HomePage() {
  const [data, setData] = useState({ temple: {current:{}}, bangkhla: {current:{}}});

    useEffect(() => {
        fetch("http://kmitl.duckdns.org:20001/", {
          headers: {
            "Content-Type": "application/json",
          }
        })
          .then(response => response.json())
          .then(data => {
            console.log(data);
            setData(data);
          })
      }, []);

    return (
        <MainLayout>
            <TitleBanner />
            <StationMenu link={"/temple"} info={textMeter} data={data.temple.current}/>
            <StationMenu link={"/bangkhla"} info={textBangkhla} data={data.bangkhla.current}/>
        </MainLayout>
    )
}

export default HomePage