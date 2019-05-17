SEC_PER_MIN = 60

def count_damaged_samples(start_time, fs, new_ones=0):
    '''We can't travel distance in vehicles without fuels, so here is the fuels

    Parameters
    ----------
    start_time : list
        A list made from struct_time, the type of the time value sequence returned by gmtime(), localtime(), and strptime(). 
        It is an object with a named tuple interface: values can be accessed by index and by attribute name.
        start_time[0, 1, 2, 3, 4, 5] = [Year, Month, Day, Hour, Minute, Seconds]

    Raises
    ------
    Nothing
        Really nothing.

    Returns
    -------
    n_skip
        Number of samples to skip.
    '''

    n_skip = 0
    
    # skip seconds
    
    if start_time[5] > 0:
        n_skip += (SEC_PER_MIN-start_time[5])*fs
        
        start_time[5] = 0
        start_time[4] += 1
        
    # skip minutes
    
    dummy = str(start_time[4])
    
    if len(dummy) == 2:
        ones = int(dummy[1])
        tens = int(dummy[0])
    else:
        ones = int(dummy[0])
        tens = 0
    
    if ones > 0:
        n_skip += (10-ones)*fs*SEC_PER_MIN
        
        if tens == 5:
            start_time[4] = 0
            start_time[3] += 1
        else:
            start_time[4] += (10-ones) #TODO 

    # TODO Perharps this part should be modified somehow. 
    if new_ones > 0: 
        start_time[4] += new_ones
        n_skip += new_ones*SEC_PER_MIN*fs

    return n_skip, start_time
