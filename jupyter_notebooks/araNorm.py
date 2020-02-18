# encoding: utf-8
'''--------------------------------------------------------------------------------
Script: Normalization class
Authors: Abdel-Rahim Elmadany and Muhammad Abdul-Mageed
Creation date: Novamber, 2018
Last update: Jan, 2019
input: text
output: normalized text
------------------------------------------------------------------------------------
Normalization functions:
- Check if text contains at least one Arabic Letter, run normalizer
- Normalize Alef and Yeh forms
- Remove Tashkeeel (diac) from Atabic text
- Reduce character repitation of > 2 characters at time
- repalce links with space
- Remove twitter username with the word USER
- replace number with NUM 
- Remove non letters or digits characters such as emoticons
------------------------------------------------------------------------------------'''
import sys
import re
#reload(sys)  # Reload does the trick!
#sys.setdefaultencoding('UTF8')

class araNorm():
	'''
		araNorm is a normalizer class for n Arabic Text
	'''
	def __init__(self):
		'''
		List of normalized characters 
		'''
		self.normalize_chars= {u"\u0622":u"\u0627", u"\u0623":u"\u0627", u"\u0625":u"\u0627", # All Araf forms to Alaf without hamza
							  u"\u0649":u"\u064A", #ALEF MAKSURA to YAH
							  u"\u0629":u"\u0647" #TEH MARBUTA to  HAH 
							}
		'''
		list of diac unicode and underscore
		'''
		self.Tashkeel_underscore_chars= {u"\u0640":"_", u"\u064E":'a', u"\u064F":'u',u"\u0650":'i',u"\u0651":'~', u"\u0652":'o', u"\u064B":'F', u"\u064C":'N', u"\u064D":'K'}
        
	def normalizeChar(self, inputText):
		'''
		step #2: Normalize Alef and Yeh forms
		'''
		norm=""
		for char in inputText:
			if char in self.normalize_chars:
				norm = norm + self.normalize_chars[char]
			else:
				norm = norm + char
		return norm
	def remover_tashkeel(self,inputText):
		'''
		step #3: Remove Tashkeeel (diac) from Atabic text
		'''
		text_without_Tashkeel=""
		for char in inputText:
			if char not in self.Tashkeel_underscore_chars:
				text_without_Tashkeel += char
		return text_without_Tashkeel
	def reduce_characters(self, inputText):
		'''
		step #4: Reduce character repitation of > 2 characters at time 
		         For example: the word 'cooooool' will convert to 'cool'
		'''
		# pattern to look for three or more repetitions of any character, including
		# newlines.
		pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
		reduced_text = pattern.sub(r"\1\1", inputText)
		return reduced_text
	def replace_links(self, inputText):
		'''
		step #5: repalce links to LINK
		         For example: http://too.gl/sadsad322 will replaced to LINK
		'''
		text = re.sub('(\w+:\/\/[ ]*\S+)','+++++++++',inputText) #LINK
		text = re.sub('\++','URL',text)
		return re.sub('(URL\s*)+',' URL ',text)
	def replace_username(self, inputText):
		'''
		step #5: Remove twitter username with the word USER
		         For example: @elmadany will replaced by space
		'''
		text = re.sub('(@[a-zA-Z0-9_]+)','USER',inputText) 
		return re.sub('(USER\s*)+',' USER ',text)
	def replace_Number(self, inputText):
		'''
		step #7: replace number with NUM 
		         For example: \d+ will replaced with NUM
		'''
		text = re.sub('[\d\.]+','NUM',inputText) 
		return re.sub('(NUM\s*)+',' NUM ',text)
	def remove_nonLetters_Digits(self, inputText):
		'''
		step #8: Remove non letters or digits characters
		         For example: emoticons...etc
		         this step is very important for w2v  and similar models; and dictionary
		'''
		p1 = re.compile('[\W_\d\s]', re.IGNORECASE | re.UNICODE)#re.compile('\p{Arabic}')
		sent = re.sub(p1, ' ', inputText)
		p1 = re.compile('\s+')
		sent = re.sub(p1, ' ', sent)
		return sent
	def run(self, text):
                normtext=""
                text=self.normalizeChar(text)
                text=self.remover_tashkeel(text)
                text=self.reduce_characters(text)
                #text=self.replace_links(text)
                #text=self.replace_username(text)
                #text=self.replace_Number(text)
                #text=self.remove_nonLetters_Digits(text)
                text = re.sub('\s+',' ', text.strip())
                text = re.sub('\s+$','', text.strip())
                normtext = re.sub('^\s+','', text.strip())
                return normtext

###############################################################
'''
Please comment below lines if used it in your package
This is ONLY an example of how to use 
'''
if __name__ == "__main__":
	norm = araNorm()
	Fwriter=open("test.norm",'w')
	# We used with open to reduce memory usage (as readline function)
	with open("test.txt",'r') as Fread:
		for line in Fread:
			cleaned_line=norm.run(line.decode('utf-8'))
			Fwriter.write(cleaned_line+"\n")
	Fwriter.close()
