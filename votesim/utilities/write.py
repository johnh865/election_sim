"""
Write tables to text.
"""
import numpy as np
from collections import OrderedDict


### String and table formatting #################################    
         

def pad_delimiters(string1, number, delimiter=',', slen=None, ):
    """
    Pad string1 to the requested number of delimiters.
    
    Parameters
    ----------
    string1 : str
        String to be assessed
    number : int
        Number of total delimiters in final output string
    delimiter : str
        String type of delimiter to use 
    """
    current_num = string1.count(delimiter)
    num_to_add = number - current_num
    
    curlen = len(string1)
    
    if num_to_add > 0:
        if slen is not None:
            len_to_add = slen - curlen
            
            spacing = int(len_to_add / num_to_add)
            xtra = len_to_add % num_to_add

            s1 = ("%" + str(spacing) + "s") % delimiter
            s1 = s1 * num_to_add + " " * xtra
            string1 += s1
        else:
            string1 += delimiter * num_to_add
    return string1

         
def format_narray(narray, fmt = '%16s', delimiter = ","):
    """
    format a 2D numpy array to a list of strings
    """
    narray = np.array(narray)
    len1 = len(narray.shape)

    if len1 == 1:
        return [format_list(narray, fmt, delimiter)]
    elif len1 == 2:
        strlist = []
        rownum = narray.shape[0]
        for i in range(rownum):
            stri = format_list(narray[i, :], fmt, delimiter)
            strlist.append(stri)
        return strlist
        

def format_list(list1, fmt = '%16s', delimiter = ","):
    """
    format list of numbers to string.
    delimiter defaults = ',' 
    """
    string1 = delimiter.join(fmt % h for h in list1) + '\n'
    return string1
    
    
    
def format_list2(list1, sfmt = '%16s', nfmt = '%16.8e', delimiter = ','):
    """
    format list of numbers or strings to a delimited string.
    
    Parameters
    ----------
    list1 : list of numbers and strings
        List to convert to string
    sfmt : str
        string formatter, ie "%16s"
    nfmt : str 
        number format, ie '%16.8e'
    delimiter : str
        list delimiter
    
    """
    outlist = []
    for h in list1:
        try:
            outlist.append(nfmt % h)
        except TypeError:
            outlist.append(sfmt % h)
    
    string1 = delimiter.join(outlist) + '\n'
    return string1
    
    
def wrap_list(list1, fmt = '%16s', delimiter = ",", maxcols = 8):
    """
    format a list and wrap by max number of columns
    """
    len1 = len(list1)
    string = ""
    for i in range(0, len1, maxcols):
        li = list1[i : i + maxcols]
        stri = format_list(li, fmt = fmt, delimiter = delimiter)
        string += stri
    return string
    

def empty_table_str(rows, columns, fmt='%16s', delimiter=','):
    """Construct empty string list for entry numbers of colnum"""
    a = [[''] * columns] * rows
    return format_narray(a, fmt=fmt, delimiter=delimiter)


#def hstack_arrays0(arrays, sfmt='%16s', nfmt='%16.8e',delimiter=','):
#    """
#    Horizontally stack arrays of varying row number and convert to strings.
#    
#    """
#    maxrows = max(len(table) for table in arrays)
#    
#    for table in arrays:
#        table = np.array(table)
#        if np.ndim(table) <= 1:
#            table = np.reshape(table, (-1, 1))
#            
#            
#        rows, cols = table.shape
#        
#        try:
#            s1 = format_narray(table, nfmt, delimiter)
#        except TypeError:
#            s1 = format_narray(table, sfmt, delimiter)
#            
#        if rows < maxrows:
#            s2 = empty_table_str(rows, cols, fmt=sfmt, delimiter=delimiter)
#            s1.extend(s2)
#        
            
        
        

def hstack_arrays(arrays, sfmt = '%16s', nfmt = '%16.8e',delimiter = ','):
    """
    Horizontally stack arrays where:
    
    Parameters
    ----------
    arrays : list/iterable
        Array-like objects
    sfmt : str
        String format (ex, '%16s')
    nfmt : str
        number format (ex, '%16.8e')
    delimiter : str
        Delimiter for columns
    
    Returns
    -------
    slist : list of string lines
    """
            
        
    s_tables = []
    for table in arrays:
        table = np.asarray(table)
        if np.ndim(table) <= 1: 
            table = np.reshape(table, (-1, 1))
        try:
            s_table = format_narray(table, nfmt, delimiter)
        except TypeError:
            s_table = format_narray(table, sfmt, delimiter)
            
        s_tables.append(s_table)
    slist = hstack_str_list(s_tables, delimiter)
    
    return slist
    

#def hstack_strings(str_lists):
#    """
#    Horizontally stack lists of strings
#    
#    arguments:
#    ------
#    str_lists : list of list of strings
#        str_lists = [str_list1, str_list2, str_list3, ...]
#    
#    str_listi : list of strings
#        List similar to output from file.readlines(). 
#        
#    
#    """
#    
#    #First get maximum row count 
#    maxrows = max(len(slist) for slist in str_lists)
#    listnum = len(str_lists)
#    max_col_lengths = []
#
#
#    #Obtain maximum line length
#    for sl in str_lists:
#        max_clen = max(len(line) for line in sl)
#        max_col_lengths.append(max_clen)
#        
#        
#    #Pad the bottoms of str lists so all lists are equal in row number
#    for i, slist in enumerate(str_lists):
#        rows = len(slist)
#        max_clen = max_col_lengths[i]
#        
#        if rows < maxrows:
#            newline = max_clen * ' '
#            padding = [newline] * (maxrows - rows)
#            slist.extend(padding)
#        
#        
#        
#        
#    
#    
#    return
#    
    



def hstack_str_list(str_lists, delimiter = ',', pad=False):
    """
    Horizontally stack lists of string in the format of:
        str_lists = [strlist1, strlist2, strlist3, ...]
        strlisti = list of strings similar to output to file.readlines()
    
    Prepare the strings for write to file, add new lines. 
    
    Parameters
    ----------
    str_lists: list of list of str
        Multiple lists of strings, each list to be horizontally stacked.
    delimiter : str
        Delimiter
        
    Returns
    -------
    out : list of str
        
    """
    
    listnum = len(str_lists)
    max_col_lengths = []
    max_col_nums = []
    
    # Strip trailing delimiter from every line in string lists. 
    for j, sl in enumerate(str_lists[:]):
        for i in range(len(sl)):
            str_lists[j][i].rstrip(delimiter)

    # Obtain max line length and max number of columns defined by delimiter.
    for sl in str_lists:
     
        max_clen = max(len(line) for line in sl)
        max_col_lengths.append(max_clen)
        
        max_columns = max(line.count(delimiter) for line in sl) + 1 
        max_col_nums.append(max_columns)       
    
    max_row_length = max(len(sl) for sl in str_lists)
    
    
    lines = []
    for i in range(max_row_length):       
        line = ""
        
        #Loop through each table to hstack
        for j in range(listnum):
            
            if j == listnum - 1:
                dnum = max_col_nums[j] - 1
            else:
                dnum = max_col_nums[j]
                
            line_len = max_col_lengths[j]
            try:
                linej = str_lists[j][i]
                linej = linej.rstrip("\n")                                     #Get rid of newlines
                linej = pad_delimiters(linej, dnum, delimiter)      #Pad out string with delimiters
                if pad:
                    linej = linej.ljust(line_len)                        #Pad out the string with spaces
                
            #Not every row in tables will have data. Pad these ones out    
            except IndexError:
                linej = pad_delimiters("", dnum, delimiter, slen=line_len)      #Pad out string with delimiters
                if pad:
                    linej = linej.ljust(line_len)                           #Pad out the string with spaces
                

            line += linej
        lines.append(line + '\n')
    return lines
                    

def get_header_names(fname, row_num, delimiter = None):
    """
    Return list of column header names as numpy array
    """
    names = []
    with open(fname) as f:
        for i in range(row_num):
            line = f.readline()
            list_ = line.split(delimiter)
            names.append(list_)
    return np.array(names, dtype = str)





class StringTable(object):
    """
    Standardardized string table for file output.
    
    Parameters
    ----------
    sfmt : str
        formatting specifier for string formatting, ex. '%16s'.
    nfmt : str
        formatting specifier for numeric formatting, ex. '%16.8e'.
    ifmt : str
        formatting specifier for integer formatting, ex. '%16d'.
    delimiter : str
        specification for delimiter. Default is ','.
    
    Usage
    -----
    Add data using self.add
    
    - data = [A x B] array of A rows of data and B columns
    - labels = [C x B] array-1d of labels for B columns.      
    - A = # of data rows
    - B = # of data columns
    - C = # of column label rows
        
        
    Add data using "add" method.
    
    """
    def __init__(self, 
                 sfmt = '%s', 
                 nfmt = '%.8e',
                 ifmt = '%d',
                 delimiter = ',',
                 header = None):
                     
        self.header_info = OrderedDict()
        self.columnlabels = []
        self.columns = []
        self.formats = []
        
        self.sfmt = sfmt
        self.nfmt = nfmt
        self.ifmt = ifmt
        self.delimiter = delimiter
        self.header = header
        
        return
    
        
    def add(self, labels, data, fmt = None):
        """
        Add data to table.
        
        Parameters
        ----------
        data : [A, B] array
            A rows of data and B columns
        labels : [C, B] array
            Labels for B columns.      
        fmt : str
            Formatting string. If None, use default number format self.nfmt
            
        * A = # of data rows
        * B = # of data columns
        * C = # of column label rows
            
        """
        data = np.asarray(data)
        self.columns.append(data)
        self.columnlabels.append(labels)
        if fmt is None:
            dtype = data.dtype.name
            if 'int' in dtype:
                fmt = self.ifmt
            else:
                fmt = self.nfmt
                
        self.formats.append(fmt)
        return
        
        
#    def add_info(self, **kwargs):
#        """Add information to header """
#        self.header_info.update(**kwargs)
#        return
#        
    
    def write(self, fname, headers=False, labels=True):
        """
        Write data to file fname. Can be file, path, or filelike.
        """
        strlist = self.string_lines(headers=headers, labels=labels)
        
        
        if not hasattr(fname, 'writelines'):
            with open(fname, 'w') as f:
                f.writelines(strlist)
        else:
            fname.writelines(strlist)
        return
        
        
    def string_lines(self, headers = False, labels = True):
        """
        Construct string data for table.
        """
        if self.header is not None:
            if self.header[-1] != '\n':
                self.header += '\n'
            strlist = [self.header]
        else:
            strlist = []
        
#        if headers:
#            s = self._build_header()
#            strlist.extend(s)
        
        if labels:
            s = self._build_labels()
            strlist.extend(s)
            
        data = self._build_data()
        strlist.extend(data)
        
        return strlist
        
    @property        
    def colnum(self):
        """Return number of columns"""
         
        colnum = 0
        for table in self.columnlabels:
            table = np.asarray(table)
            if np.ndim(table) <= 1:
                table = np.reshape(table, (1, -1))
            colnum += table.shape[1]
        return colnum
        
        
    def _build_data(self):
        
        s_tables = []
        for table, fmt in zip(self.columns, self.formats):
            
            table = np.array(table)
            if np.ndim(table) <= 1: 
                table = np.reshape(table, (-1, 1))
            try:
                s_table = format_narray(table, fmt, self.delimiter)
            except TypeError:
                s_table = format_narray(table, self.sfmt, self.delimiter)
                
            s_tables.append(s_table)
        slist = hstack_str_list(s_tables, self.delimiter)
        return slist
        
    
    def _build_labels(self):
        
        s_tables = []
        for table in self.columnlabels:
            table = np.array(table)
            if np.ndim(table) <= 1: 
                table = np.reshape(table, (1, -1))
                
            s_table = format_narray(table, self.sfmt, self.delimiter)
            s_tables.append(s_table)

        slist = hstack_str_list(s_tables, self.delimiter)            
            
        return slist
    
        
        
#        
#    def _build_header(self):
#        
#        list1 = []
#        for key, val in self.header_info.items():
#            fkey = key + '='
#            try:
#                fval = self.sfmt % val
#            except TypeError:
#                fval = self.nfmt % val
#            list1.append(fkey)
#            list1.append(fval)
#        
#        slist = wrap_list(list1, self.sfmt, self.delimiter, maxcols = self.colnum)
#        return slist
        

class ReadTable(object):
    """
    Read file constructed using StringTable.
    
    All data is read as str and as numpy arrays. All data is indexed as both
    ether a numpy array or as a using keys found from the labels.
    
    
    Parameters
    ----------
    f : file-like or iterable
        file or iterable with text data
    labelrows : list or None
        List of column row numbers that have column label information. 
        By default, this is set to None to use only 0th row. 
    delimiter : str
        Table delimiter
    
    """
    
    def __init__(self, f, labelrows=None, delimiter=','):
        self.lines = list(f)
        
        if labelrows is None:
            labelrows = [0]
        self.labelrows = labelrows
        self.delimiter = delimiter
        self.labels = self._read_labels()
        self.data = self._read_data()
        return
    
    
    def _read_labels(self):
        """Read columnn headers/labels"""
        lines = self.lines
        hlines = [lines[i] for i in self.labelrows]
        
        header = np.genfromtxt(hlines, delimiter=',', dtype=str)
        header = np.core.defchararray.strip(header)
        
        add = np.core.defchararray.add
        
        header1 = add(header, '\n')
        
        newheader = header1[0]
        for h in header1[1:]:
            newheader = add(newheader, h)
        
        newheader = [s.strip() for s in newheader]
        #newheader = [s.replace('\n', '.') for s in newheader]
        
        return newheader
    
    
    def _read_data(self):
        """Read tabular data"""
        lines = self.lines
        start = max(self.labelrows) + 1
        d = np.genfromtxt(lines[start:], 
                          delimiter=self.delimiter,
                          dtype=str)
        return d
    
    
    def dict(self):
        d = {}
        for i, l in enumerate(self.labels):
            d[l] = self.data[:, i]
        return d
    
    
    def find(self, key):
        """
        Find keyword within labels. Return columns with keyword
        
        returns:
        ------
        array of shape (a, b)
        """
        d = []
        for k in self.labels:
            if key in k:
              d.append(self[k])
        return np.column_stack(d)
    
    
    def __getitem__(self, key):
        try:
            return self.dict[key]
        except:
            return self.data[key]
            
    
            
    
    
        
        
if __name__ == '__main__':
    
    t = pad_delimiters('bob, joe, tim', 5, slen=50)
    
    
    a = np.ones((10,3))
    alabels = 'joe','bob','tim'
    
    b=  np.zeros((20,4))
    blabels = 'p1','p2','p3','p4'
    
    c = np.random.rand(40,5)
    clabels = [['bob','john','cid','doe'],['b','j','c','d']]
    
    s = StringTable()
    s.add(alabels, c)
    s.add(blabels, b)
    s.add(clabels, a)
    s.write('test-file.txt')
    pass


