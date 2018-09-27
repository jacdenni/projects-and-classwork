function convertToRoman(num) {
	var numCopy = num;//don't modify stuff that gets passed in if we can help it!

	var romanNum = ''; //our eventual output

	while(numCopy >= 1000) {
		romanNum += 'M';	
		numCopy -= 1000;
	}
	while(numCopy >= 500) {
		if (numCopy >= 900) {
			romanNum += 'CM'
			numCopy -= 900;
		}
		else {
			romanNum += 'D';
			numCopy -= 500;
		}
	}
	while(numCopy >= 100) {
		if (numCopy >= 400) {
			romanNum += 'CD'
			numCopy -= 400;
		}
		else {
			romanNum += 'C';
			numCopy -= 100;
		}
	}
	while(numCopy >= 50) {
		if (numCopy >= 90) {
			romanNum += 'XC'
			numCopy -= 90;
		}
		else {
			romanNum += 'L';
			numCopy -= 50;
		}
	}
	while(numCopy >= 10) {
		if (numCopy >= 40) {
			romanNum += 'XL'
			numCopy -= 40;
		}
		else {
			romanNum += 'X';
			numCopy -= 10;
		}
	}
	while(numCopy >= 5) {
		if (numCopy >= 9) {
			romanNum += 'IX'
			numCopy -= 9;
		}
		else {
			romanNum += 'V';
			numCopy -= 5;
		}
	}
	while(numCopy >= 1) {
		if (numCopy >= 4) {
			romanNum += 'IV'
			numCopy -= 4;
		}
		else {
			romanNum += 'I';
			numCopy -= 1;
		}
	}
	return romanNum;
}

convertToRoman(3999);