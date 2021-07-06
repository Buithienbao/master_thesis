### Exercise 1

def sumEvenFib(numth=100):

	"""
    Calculate sum of numth even-valued Fibonacci numbers

    Parameters
    ----------
    numth : int
        number of even-valued Fibonacci numbers

    Returns
    -------
    total : int
    	sum of numth even-valued Fibonacci numbers
	"""

	#Initialze the first two even Fibonacci number
	f1 = 0
	f2 = 2
	
	#Sum of these numbers
	total = f1+f2
	
	#Number of even Fibonacci number counter
	count = 2

	while (count!=numth): #Stop the loop when sum of numth even number are calculated

		#Calculate the next even Fibonacci number
		f3 = 4*f2 + f1
		
		#Add this value to the sum
		total += f3

		#Increment counter by 1
		count += 1
		
		#Move to the next even number
		f1 = f2
		f2 = f3

	return total

### Exercise 2


if __name__ == '__main__':

	sumEvenFib()