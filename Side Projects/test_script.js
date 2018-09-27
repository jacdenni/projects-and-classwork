function checkCashRegister(price, cash, cid) {
  var cashDrawerCopy = cid.map((arr) => arr.slice());//create a deep copy of the cash in the drawer

  //Calculate the change needed and create our change array
  var changeNeeded = cash - price;
  var changeArr = [];
  
  //Get the change from the drawer, checking what we need at each denomination

  if (changeNeeded >= 100) {
    //see how many hundreds we need
    var oneHundreds = ["ONE HUNDRED", 0] // hundreds array to add to changeArr
    while (changeNeeded >= 100 && cashDrawerCopy[8][1] > 0) {
      cashDrawerCopy[8][1] -= 100;
      oneHundreds[1] += 100;
      changeNeeded -= 100;
    }
    if (oneHundreds[1] > 0) {
    	changeArr.push(oneHundreds);
	}
  }

  if (changeNeeded >= 20) {
    //see how many twenties we need
    var twenties = ["TWENTY", 0] // twenties array to add to changeArr
    while (changeNeeded >= 20 && cashDrawerCopy[7][1] > 0) {
      cashDrawerCopy[7][1] -= 20;
      twenties[1] += 20;
      changeNeeded -= 20;
    }
    if (twenties[1] > 0) {
    	changeArr.push(twenties);
	}
  }

  if (changeNeeded >= 10) {
    //see how many tens we need
    var tens = ["TEN", 0] // tens array to add to changeArr
    while (changeNeeded >= 10 && cashDrawerCopy[6][1] > 0) {
      cashDrawerCopy[6][1] -= 10;
      tens[1] += 10;
      changeNeeded -= 10;
    }
    if (tens[1] > 0) {
    	changeArr.push(tens);
    }
  }

  if (changeNeeded >= 5) {
    //see how many fives we need
    var fives = ["FIVE", 0] // fives array to add to changeArr
    while (changeNeeded >= 5 && cashDrawerCopy[5][1] > 0) {
      cashDrawerCopy[5][1] -= 5;
      fives[1] += 5;
      changeNeeded -= 5;
    }
    if (fives[1] > 0) {
    	changeArr.push(fives);
	}
  }

  if (changeNeeded >= 1) {
    //see how many ones we need
    var ones = ["ONE", 0] // ones array to add to changeArr
    while (changeNeeded >= 1 && cashDrawerCopy[4][1] > 0) {
      cashDrawerCopy[4][1] -= 1;
      ones[1] += 1;
      changeNeeded -= 1;
    }
    if (ones[1] > 0) {
    	changeArr.push(ones);
	}
  }

  if (changeNeeded >= 0.25) {
    //see how many quarters we need
    var quarters = ["QUARTER", 0] // quarters array to add to changeArr
    while (changeNeeded >= 0.25 && cashDrawerCopy[3][1] > 0) {
      cashDrawerCopy[3][1] -= 0.25;
      quarters[1] += 0.25;
      changeNeeded -= 0.25;
    }
    if (quarters[1] > 0) {
    	changeArr.push(quarters);
	}
  }

  if (changeNeeded >= 0.1) {
    //see how many dimes we need
    var dimes = ["DIME", 0] // dimes array to add to changeArr
    while (changeNeeded >= 0.1 && cashDrawerCopy[2][1] > 0) {
      cashDrawerCopy[2][1] -= 0.1;
      dimes[1] += 0.1;
      changeNeeded -= 0.1;
    }
    if (dimes[1] > 0) {
    	changeArr.push(dimes);
	}
  }

  if (changeNeeded >= 0.05) {
    //see how many nickels we need
    var nickels = ["NICKEL", 0] // nickels array to add to changeArr
    while (changeNeeded >= 0.05 && cashDrawerCopy[1][1] > 0) {
      cashDrawerCopy[1][1] -= 0.05;
      nickels[1] += 0.05;
      changeNeeded -= 0.05;
    }
    if (nickels[1] > 0) {
    	changeArr.push(nickels);
	}
  }
  
  if (changeNeeded >= 0.01) {
  	//multiply everything by 100 to get rid of errors
  	changeNeeded *= 100;
  	cashDrawerCopy[0][1] *= 100;
    //see how many pennies we need
    var pennies = ["PENNY", 0] // pennies array to add to changeArr
    while (changeNeeded > 0) {
      cashDrawerCopy[0][1] -= 1;
      pennies[1] += 1;
      changeNeeded -= 1;
    }

    if (cashDrawerCopy[0][1] < 0) {
    	return {status: 'INSUFFICIENT_FUNDS', change: []};
    }

    cashDrawerCopy[0][1] /= 100;
    pennies[1] /= 100;
    if (pennies[1] > 0) {
    	changeArr.push(pennies);
	}
  }

  //check to see if we emptied the cash drawer
  for (let i = 0; i < cashDrawerCopy.length; ++i) {
  	if (cashDrawerCopy[i][1] > 0) {
  		return {status: 'OPEN', change: changeArr};
  	}
  }
  //should only execute if all the drawers are equal to 0
  return {status: 'CLOSED', change: cid};
}

// Example cash-in-drawer array:
// [["PENNY", 1.01],
// ["NICKEL", 2.05],
// ["DIME", 3.1],
// ["QUARTER", 4.25],
// ["ONE", 90],
// ["FIVE", 55],
// ["TEN", 20],
// ["TWENTY", 60],
// ["ONE HUNDRED", 100]]

var result = checkCashRegister(19.5, 20, [["PENNY", 0.5], ["NICKEL", 0], ["DIME", 0], ["QUARTER", 0], ["ONE", 0], ["FIVE", 0], ["TEN", 0], ["TWENTY", 0], ["ONE HUNDRED", 0]]);

debug(`status: ${result.status}\n${result.change}`);