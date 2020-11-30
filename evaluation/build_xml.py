'''
This script converts a set of text files to an XML file.

It's used to generate lexical chains with get_senses.py and lexCH.py in the lexical chain submodule.

Nov 29th, 2020

Benjamin LeBrun
'''
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
import xml.etree.ElementTree as ET
import os, re
import argparse

# tokenize and pos tag
# I'm gonna change this, we're currently tokenizing twice (here and cohesion.py) 
# which is slow for no reason
import spacy
tokenizer = spacy.load("en_core_web_lg", disable=["parser", "ner", "textcat"])
tokenizer.add_pipe(tokenizer.create_pipe('sentencizer'))

def add_document(motherNode, segments, docId):
	''' adds document to XML file'''
	docNode = SubElement(motherNode, 'doc')
	docNode.set('docid', docId)
	for segment, i in zip(segments, [i for i in range(len(segments))]):
		seg = add_segment(docNode, i, segment)
	return docNode

def add_segment(motherNode, segId, text):
	''' adds segments to document node '''
	segment = SubElement(motherNode, 'seg')
	segment.set('segid', str(segId))
	segment.text = text
	return segment

def build_xml_doc(setId, file_dir, srclang='English'):
	''' builds xml doc from a list of files '''
	top = Element('tgeval')
	top.set('fileid','test')
	mother = SubElement(top, 'srcset')
	mother.set('setid', setId)
	mother.set('srclang', srclang)

	files = os.listdir(file_dir)
	for file in files:
		with open(file_dir+file, 'r') as f:
			spacy_doc=tokenizer(f.read())
			sentences=[re.sub('\n', '', s.text) for s in spacy_doc.sents]
			doc = add_document(mother, sentences, file)

	return top

def write_doc(doc, name):
	with open(name, 'w') as f:
		f.write(ET.tostring(doc).decode())

def main():
	parser = argparse.ArgumentParser('Create XML file.')
	parser.add_argument('--input-files', help='directory containing input files', required=True)
	parser.add_argument('--output-dest', help='output destination', required=True)
	parser.add_argument('--set-id', help='name for this set of files', required=True)

	args = parser.parse_args()

	doc = build_xml_doc(args.set_id, args.input_files)
	write_doc(doc, args.output_dest)
	
if __name__=='__main__':
	main()