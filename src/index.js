import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import HomePage from './pages/HomePage';
import BangKhla from './pages/BangKhla';
import Temple from './pages/Temple'
import reportWebVitals from './reportWebVitals';
import {createBrowserRouter, RouterProvider} from "react-router-dom";
import socketIOClient from "socket.io-client";
import { SouthEast } from '@mui/icons-material';


const socket = socketIOClient("http://kmitl.duckdns.org:20001/hourly");


const router = createBrowserRouter([
  // {
  //   path: "/",
  //   element: <App />,
  // },
  {
    path: "/",
    element: <HomePage />,
  },
  {
    path: "/temple",
    element: <Temple socket={socket}/>,
  },
  {
    path: "/bangkhla",
    element: <BangKhla socket={socket}/>,
  }
]);

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    {/* <App /> */}
    <RouterProvider router={router} />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
