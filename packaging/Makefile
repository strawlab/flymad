GITREV:=$(shell git rev-parse --short --verify HEAD)
VERSION:=$(shell python -c "import roslib; roslib.load_manifest('flymad'); import flymad; print flymad.__version__")
TMPDIR:=$(shell mktemp --directory)
FLYMAD_BASE_DIR:=..

default: release

snapshot: flymad-snapshot-$(GITREV).zip flymad-snapshot-$(GITREV).tar.gz

release: flymad-release-$(VERSION).zip flymad-release-$(VERSION).tar.gz

# snapshot targets -------------------------------------
flymad-snapshot-$(GITREV).zip: flymad-snapshot-$(GITREV)
	zip -9 -r $@ $<

flymad-snapshot-$(GITREV).tar.gz: flymad-snapshot-$(GITREV)
	tar -czf $@ $<

flymad-snapshot-$(GITREV): flymad-packaging-dir
	rm -rf $@
	cp -a $< $@

# release targets -------------------------------------
flymad-release-$(VERSION).zip: flymad-release-$(VERSION)
	zip -9 -r $@ $<

flymad-release-$(VERSION).tar.gz: flymad-release-$(VERSION)
	tar -czf $@ $<

flymad-release-$(VERSION): flymad-packaging-dir
	rm -rf $@
	cp -a $< $@

# the actual packaged file -------------------------------------
flymad-packaging-dir: install-flymad-on-ubuntu-12.04-amd64.sh tarball-readme.md $(FLYMAD_BASE_DIR)/doc/mechanical_drawings $(FLYMAD_BASE_DIR)/flymad_micro/PCB/shield_10b.zip
	cp -a $^ $(TMPDIR)
	mv $(TMPDIR)/tarball-readme.md $(TMPDIR)/README
	rm -rf $@
	mv $(TMPDIR) $@

# installer script -------------------------------------
install-flymad-on-ubuntu-12.04-amd64.sh: make-redist-sh.sh tarball-readme.md help-header.txt install-flymad.sh
	./make-redist-sh.sh
