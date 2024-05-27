import BlueSaltWater from '../icons/salt-water/03/blue-salt-water.png'
import YellowSaltWater from '../icons/salt-water/03/yellow-salt-water.png'
import RedSaltWater from '../icons/salt-water/03/red-salt-water.png'

function roundTo(n, place) {    
    return +(Math.round(n + "e+" + place) + "e-" + place);
}

export function mapImg(value) {
    const roundedValue = roundTo(value, 2);
    if (roundedValue > 2.0){
        return RedSaltWater;
    }
    else if(roundedValue > .5){
        return YellowSaltWater;
    }
    else{
        return BlueSaltWater;
    }
}