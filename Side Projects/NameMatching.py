import numpy as np
from os import system

np.random.seed(0)


def clearScreen():
	_ = system('clear')

def getNames():
	playersList = []
	numPlayers = int(input("How many names would you like to put in the hat? "))
	for player in range(1, numPlayers + 1):
		playersList.append(input("Enter name of player {}: ".format(player)))
	return playersList

def matchPlayers(playersList):
	numPlayers = len(playersList)
	mutablePlayersList = playersList.copy() #make a copy that we can remove players from without modifying the original players
	matchedPlayers = {}
	for player in playersList:
		partner = mutablePlayersList[np.random.randint(0, numPlayers)]
		
		while (partner == player): #if partner = player, roll again!
			partner = mutablePlayersList[np.random.randint(0, numPlayers)]

		matchedPlayers[player] = partner
		mutablePlayersList.remove(partner)
		numPlayers -= 1
	return matchedPlayers






def main():
	playersList = getNames()
	matchedPlayers = matchPlayers(playersList)
	response = ''
	while (response != 'quit'):
		response = input('Enter the name of the player whose partner you want to see (enter \'quit\' to exit): ')
		while ((response not in matchedPlayers) and (response != 'quit')):
			response = input('I\'m sorry, that person is not in the game. Please enter another name: ')
		if (response != 'quit'):
			print('{0} was matched with {1}'.format(response, matchedPlayers[response]))
			checkClear = input('Do you want to clear the screen? (Y/N): ')
			if (checkClear == 'Y' or checkClear == 'y'):
				clearScreen()
	print('Goodbye.')


main()

