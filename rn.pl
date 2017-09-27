use v5.10;
my @list = glob '*.mp4';

for my $fn(@list) {
	my $filename = $fn;
	$filename =~ s/[^\w -\.]/_/g;
	$filename =~ s/\'//g;
	$filename =~ s/\+//g;
	$filename =~ s/\.+?mp4$/.mp4/g;
	rename($fn, $filename);
}
