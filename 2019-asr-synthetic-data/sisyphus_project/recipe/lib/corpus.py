__all__ = ['NamedEntity', 'CorpusSection', 'Corpus', 'Recording', 'Segment', 'Speaker']

import collections
import gzip
import os
import re
import xml.sax as sax
import xml.sax.saxutils as saxutils
import xml.etree.ElementTree as ET


class NamedEntity:
  def __init__(self):
    super().__init__()
    self.name = None


class CorpusSection:
  def __init__(self):
    super().__init__()
    self.speaker_name    = None
    self.default_speaker = None

    self.speakers = collections.OrderedDict()


class CorpusParser(sax.handler.ContentHandler):
  """
  This classes methods are called by the sax-parser whenever it encounters an event in the xml-file
  (tags/characters/namespaces/...). It uses a stack of elements to remember the part of the corpus that
  is currently beeing read.
  """

  def __init__(self, corpus, path):
    super().__init__()

    self.elements = [corpus]  # stack of objects to store the element of the corpus that is beeing read
    self.path     = path      # path of the parent corpus (needed for include statements)
    self.chars    = ''        # buffer for character events, it is reset whenever a new element starts

  def startElement(self, name, attrs):
    e = self.elements[-1]
    if name == 'corpus':
      assert len(self.elements) == 1, "<corpus> may only occur as the root element"
      e.name = attrs['name']
    elif name == 'subcorpus':
      assert isinstance(e, Corpus), "<subcorpus> may only occur within a <corpus> or <subcorpus> element"
      subcorpus = Corpus()
      subcorpus.name = attrs['name']
      subcorpus.parent_corpus = e
      e.subcorpora.append(subcorpus)
      self.elements.append(subcorpus)
    elif name == 'include':
      assert isinstance(e, Corpus), "<include> may only occur within a <corpus> or <subcorpus> element"
      path = os.path.join(os.path.dirname(self.path), attrs['file'])
      c = Corpus()
      c.load(path)
      if c.name != e.name:
        print('Warning: included corpus (%s) has a different name than the current corpus (%s)' % (c.name, e.name))
      e.subcorpora.extend(c.subcorpora)
      e.recordings.extend(c.recordings)
      e.speakers.update(c.speakers)
    elif name == 'recording':
      assert isinstance(e, Corpus), "<recording> may only occur within a <corpus> or <subcorpus> element"
      rec = Recording()
      rec.name  = attrs['name']
      rec.audio = attrs['audio']
      rec.corpus = e
      e.recordings.append(rec)
      self.elements.append(rec)
    elif name == 'segment':
      assert isinstance(e, Recording), "<segment> may only occur within a <recording> element"
      seg = Segment()
      seg.name  = attrs.get('name', str(len(e.segments) + 1))
      seg.start = float(attrs.get('start', '0.0'))
      seg.end   = float(attrs.get('end', '0.0'))
      seg.track = int(attrs['track']) if 'track' in attrs else None
      seg.recording = e
      e.segments.append(seg)
      self.elements.append(seg)
    elif name == 'speaker-description':
      assert isinstance(e, CorpusSection), "<speaker-description> may only occur within a <corpus>, <subcorpus> or <recording>"
      speaker = Speaker()
      speaker.name = attrs.get('name', None)
      if speaker.name is not None:
        e.speakers[speaker.name] = speaker
      else:
        e.default_speaker = speaker
      self.elements.append(speaker)
    elif name == 'speaker':
      assert isinstance(e, (CorpusSection, Segment)), "<speaker> may only occur within a <corpus>, <subcorpus>, <recording> or <segment>"
      e.speaker_name = attrs['name']
    self.chars = ''

  def endElement(self, name):
    e = self.elements[-1]

    if name == 'orth':
      assert isinstance(e, Segment)
      # we do some processing of the text that goes into the orth tag to get a nicer formating, some corpora may have
      # multiline content in the orth tag, but to keep it that way might not be consistent with the indentation during
      # writing, thus we remove multiple spaces and newlines
      text = self.chars.strip()
      text = re.sub(' +', ' ', text)
      text = re.sub('\n', '', text)
      e.orth = text
    elif isinstance(e, Speaker) and name != 'speaker-description':
      # we allow all sorts of elements within a speaker description
      e.attribs[name] = self.chars.strip()

    if name in ['corpus', 'subcorpus', 'recording', 'segment', 'speaker-description']:
      self.elements.pop()

  def characters(self, characters):
    self.chars += characters


class Corpus(NamedEntity, CorpusSection):
  """
  This class represents a corpus in the Bliss format. It is also used to represent subcorpora when the parent_corpus
  attribute is set. Corpora with include statements can be read but are written back as a single file.
  """

  def __init__(self):
    super().__init__()

    self.parent_corpus = None

    self.subcorpora = []
    self.recordings = []

  def segments(self):
    """
    @return : an iterator over all segments within the corpus
    """
    for r in self.recordings:
      yield from r.segments
    for sc in self.subcorpora:
      yield from sc.segments()

  def all_recordings(self):
    """
    :rtype: Iterator[Recording]
    """
    yield from self.recordings
    for sc in self.subcorpora:
      yield from sc.all_recordings()

  def remove_recording(self, recording):
    to_delete = []
    for idx, r in enumerate(self.recordings):
      if r is recording or r == recording or r.name == recording:
        to_delete.append(idx)
    for idx in reversed(to_delete):
      del self.recordings[idx]
    for sc in self.subcorpora:
      sc.remove_recording(recording)

  def add_recording(self, recording):
    assert isinstance(recording, Recording)
    recording.corpus = self
    self.recordings.append(recording)

  def add_subcorpus(self, corpus):
    assert isinstance(corpus, Corpus)
    corpus.parent_corpus = self
    self.subcorpora.append(corpus)

  def add_speaker(self, speaker):
    assert isinstance(speaker, Speaker)
    self.speakers[speaker.name] = speaker

  def fullname(self):
    if self.parent_corpus is not None:
      return self.parent_corpus.fullname() + '/' + self.name
    else:
      return self.name

  def speaker(self, speaker_name, default_speaker):
    if speaker_name is None:
      speaker_name = self.speaker_name
    if speaker_name in self.speakers:
      return self.speakers[speaker_name]
    else:
      if default_speaker is None:
        default_speaker = self.default_speaker
      if self.parent_corpus is not None:
        return self.parent_corpus.speaker(speaker_name, default_speaker)
      else:
        return default_speaker

  def load(self, path):
    """
    :param str path:
    """
    open_fun = gzip.open if path.endswith('.gz') else open

    with open_fun(path, 'rt') as f:
      handler = CorpusParser(self, path)
      sax.parse(f, handler)

  def dump(self, path):
    """
    :param str path:
    """
    open_fun = gzip.open if path.endswith('.gz') else open

    with open_fun(path, 'wt') as f:
      f.write('<?xml version="1.0" encoding="utf8"?>\n')
      self._dump_internal(f)

  def _dump_internal(self, out, indentation=''):
    if self.parent_corpus is None:
      out.write('<corpus name="%s">\n' % self.name)
    else:
      out.write('%s<subcorpus name="%s">\n' % (indentation, self.name))

    for s in self.speakers.values():
      s.dump(out, indentation + '  ')

    for r in self.recordings:
      r.dump(out, indentation + '  ')

    for sc in self.subcorpora:
      sc._dump_internal(out, indentation + '  ')

    if self.parent_corpus is None:
      out.write('</corpus>\n')
    else:
      out.write('%s</subcorpus>\n' % (indentation,))


class Recording(NamedEntity, CorpusSection):
  def __init__(self):
    super().__init__()
    self.audio = None
    self.corpus = None
    self.segments = [] # type: List[Segment]

  def fullname(self):
    return self.corpus.fullname() + '/' + self.name

  def speaker(self, speaker_name=None):
    if speaker_name is None:
      speaker_name = self.speaker_name
    if speaker_name in self.speakers:
      return self.speakers[speaker_name]
    else:
      return self.corpus.speaker(speaker_name, self.default_speaker)

  def dump(self, out, indentation=''):
    out.write('%s<recording name="%s" audio="%s">\n' % (indentation, self.name, self.audio))

    for s in self.speakers.values():
      s.dump(out, indentation + '  ')

    for s in self.segments:
      s.dump(out, indentation + '  ')

    out.write('%s</recording>\n' % indentation)

  def add_segment(self, segment):
    assert isinstance(segment, Segment)
    segment.recording = self
    self.segments.append(segment)


class Segment(NamedEntity):
  def __init__(self):
    super().__init__()
    self.start        = 0.0
    self.end          = 0.0
    self.track        = None
    self.orth         = None
    self.speaker_name = None

    self.recording = None

  def fullname(self):
    return self.recording.fullname() + '/' + self.name

  def speaker(self):
    return self.recording.speaker(self.speaker_name)

  def dump(self, out, indentation=''):
    has_child_element = self.orth is not None or self.speaker_name is not None
    t = ' track="%d"' % self.track if self.track is not None else ''
    new_line = '\n' if has_child_element else ''
    out.write('%s<segment name="%s" start="%.4f" end="%.4f"%s>%s' % (indentation, self.name, self.start, self.end, t, new_line))
    if self.speaker_name is not None:
      out.write('%s  <speaker name="%s"/>\n' % (indentation, self.speaker_name))
    if self.orth is not None:
      out.write('%s  <orth> %s </orth>\n' % (indentation, saxutils.escape(self.orth)))
    if has_child_element:
      out.write('%s</segment>\n' % indentation)
    else:
      out.write('</segment>\n')


class Speaker(NamedEntity):
  def __init__(self):
    super().__init__()
    self.attribs = {}

  def dump(self, out, indentation=''):
    out.write('%s<speaker-description%s>' % (indentation, (' name="%s"' % self.name) if self.name is not None else ''))
    if len(self.attribs) > 0:
      out.write('\n')
    for k, v in self.attribs.items():
      out.write('%s  <%s>%s</%s>\n' % (indentation, k, v, k))
    out.write('%s</speaker-description>\n' % (indentation if len(self.attribs) > 0 else ''))


class SegmentMap(object):

  def __init__(self):
    self.map_entries = []

  def load(self, path):
    """
    :param str path:
    """
    open_fun = gzip.open if path.endswith('.gz') else open

    with open_fun(path, 'rb') as f:
      for event, elem in ET.iterparse(f, events=('start',)):
        if elem.tag == "map-item":
          elem = elem # type: etree.Element
          item = SegmentMapItem()
          item.key = elem.attrib['key']
          item.value = elem.attrib['value']
          self.map_entries.append(item)

  def dump(self, path):
    """
    :param str path:
    """
    open_fun = gzip.open if path.endswith('.gz') else open

    with open_fun(path, 'wt') as f:
      f.write('<?xml version="1.0" encoding="utf8"?>\n')
      f.write('<segment-key-map>\n')

      for s in self.map_entries:
        s.dump(f)

      f.write('</segment-key-map>\n')

class SegmentMapItem(object):

  def __init__(self):
    self.key = None
    self.value = None

  def dump(self, out,):
    out.write('<map-item key="%s" value="%s" />\n' % (self.key, self.value))
