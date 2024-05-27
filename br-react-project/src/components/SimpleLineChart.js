import React, { PureComponent } from 'react';
import { LineChart, Line, XAxis, YAxis, Label, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';


export default class SimpleLineChart extends PureComponent {
  constructor(props) {
    super(props);
    this.state = { data: props.data, graph: props.graph };
  }

  componentDidUpdate(prevProps) {
    if (prevProps.data !== this.props.data) {
      this.setState({ data: this.props.data });
    }
  }


  render() {
    const { timeRange } = this.props;
    const { data, graph } = this.state;
    const {dataKeyX} = this.props;
    
    const slicedData = ((data.length >= timeRange) ? (data.slice(-timeRange)) : data);

    const gl_actual_min = Math.min(...slicedData.map(item => item.gl_actual));
    const gl_actual_max = Math.max(...slicedData.map(item => item.gl_actual));
    
    const gl_pred_min = Math.min(...slicedData.map(item => item.gl_pred));
    const gl_pred_max = Math.max(...slicedData.map(item => item.gl_pred));

    var minVar = "auto", maxVar = "auto";

    if( gl_actual_min < 0 ){
      minVar = gl_pred_min -.5;
    }
    else if( gl_actual_min < gl_pred_min ){
      minVar = gl_actual_min -.5;
    }

    if( gl_actual_max > gl_pred_max ){
      maxVar = gl_actual_max +.5;
    }

    const lines = this.state.graph.line.map((line, index) => (
      <Line key={index}
        type={line.type}
        dataKey={line.dataKey}
        stroke={line.color}
        name={line.linename}
        dot={(slicedData.length <= 24) ? true : false}
      />
    ));

    const refLines = this.state.graph.refLineY.map((line, index) => (
      <ReferenceLine key={index} y={line.y} stroke={line.color} strokeDasharray={line.linestyle} />
    ));


    return (
      <ResponsiveContainer width="100%" height="125%">
        <LineChart
          width={500}
          height={300}
          data={slicedData}
          margin={{
            top: 0,
            right: 0,
            left: 10,
            bottom: 25,
          }}
        >
          {/* <CartesianGrid strokeDasharray="3 3" /> */}
          <XAxis dataKey={dataKeyX} tickCount={slicedData.length / 4} >
            <Label value="วันที่" position='bottom' style={{fontSize:"100%"}} offset={10}/>
          </XAxis>
          <YAxis type="number" domain={[minVar, maxVar]} allowDataOverflow={true}>
            <Label value="ความเค็ม (g/l)" angle={270} position='left' style={{ textAnchor: 'middle', fontSize:"110%"}} offset={-1} />
          </YAxis>
          <Legend layout="horizontal" verticalAlign="top" align="center" />
          <Tooltip />
          
          {/* {refLines} */}
          {lines}
        </LineChart>
      </ResponsiveContainer>
    );
  }
}
